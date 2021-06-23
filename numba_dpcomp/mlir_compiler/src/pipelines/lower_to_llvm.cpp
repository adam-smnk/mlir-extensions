// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pipelines/lower_to_llvm.hpp"

#include <mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

#include "plier/dialect.hpp"

#include "plier/transforms/func_utils.hpp"

#include "base_pipeline.hpp"
#include "plier/compiler/pipeline_registry.hpp"

#include "plier/utils.hpp"

namespace {
mlir::LowerToLLVMOptions getLLVMOptions(mlir::MLIRContext &context) {
  static llvm::DataLayout dl = []() {
    llvm::InitializeNativeTarget();
    auto triple = llvm::sys::getProcessTriple();
    std::string err_str;
    auto target = llvm::TargetRegistry::lookupTarget(triple, err_str);
    if (nullptr == target) {
      plier::report_error(llvm::Twine("Unable to get target: ") + err_str);
    }
    llvm::TargetOptions target_opts;
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        triple, llvm::sys::getHostCPUName(), "", target_opts, llvm::None));
    return machine->createDataLayout();
  }();
  mlir::LowerToLLVMOptions opts(&context);
  opts.dataLayout = dl;
  opts.useBarePtrCallConv = false;
  opts.emitCWrappers = false;
  opts.allocLowering = mlir::LowerToLLVMOptions::AllocLowering::None;
  return opts;
}

llvm::Optional<mlir::Type> convertTuple(mlir::LLVMTypeConverter &converter,
                                        mlir::TupleType tuple) {
  if (tuple.getTypes().empty()) {
    return mlir::LLVM::LLVMStructType::getLiteral(tuple.getContext(),
                                                  llvm::None);
  }

  auto unitupleType = [&]() -> mlir::Type {
    auto types = tuple.getTypes();
    assert(!types.empty());
    auto elemType = types.front();
    types = types.drop_front();
    if (llvm::all_of(types, [&](auto t) { return t == elemType; })) {
      return elemType;
    }
    return nullptr;
  }();

  auto count = static_cast<unsigned>(tuple.getTypes().size());
  if (unitupleType) {
    auto newType = converter.convertType(unitupleType);
    if (!newType) {
      return llvm::None;
    }
    return mlir::LLVM::LLVMArrayType::get(unitupleType, count);
  }
  llvm::SmallVector<mlir::Type> newTypes;
  newTypes.reserve(count);
  for (auto type : tuple.getTypes()) {
    auto newType = converter.convertType(type);
    if (!newType) {
      return llvm::None;
    }
    newTypes.emplace_back(newType);
  }

  return mlir::LLVM::LLVMStructType::getLiteral(tuple.getContext(), newTypes);
}

void populateToLLVMAdditionalTypeConversion(
    mlir::LLVMTypeConverter &converter) {
  converter.addConversion([&converter](mlir::TupleType type) {
    return convertTuple(converter, type);
  });
  auto voidPtrType = mlir::LLVM::LLVMPointerType::get(
      mlir::IntegerType::get(&converter.getContext(), 8));
  converter.addConversion(
      [voidPtrType](plier::NoneType) -> llvm::Optional<mlir::Type> {
        return voidPtrType;
      });
  converter.addConversion(
      [](mlir::IntegerType type) -> llvm::Optional<mlir::Type> {
        if (!type.isSignless()) {
          return mlir::IntegerType::get(type.getContext(), type.getWidth());
        }
        return llvm::None;
      });
}

struct LLVMTypeHelper {
  LLVMTypeHelper(mlir::MLIRContext &ctx) : type_converter(&ctx) {
    populateToLLVMAdditionalTypeConversion(type_converter);
  }

  mlir::Type i(unsigned bits) {
    return mlir::IntegerType::get(&type_converter.getContext(), bits);
  }

  mlir::Type ptr(mlir::Type type) {
    assert(static_cast<bool>(type));
    auto ll_type = type_converter.convertType(type);
    assert(static_cast<bool>(ll_type));
    return mlir::LLVM::LLVMPointerType::get(ll_type);
  }

  mlir::MLIRContext &get_context() { return type_converter.getContext(); }

  mlir::LLVMTypeConverter &get_type_converter() { return type_converter; }

private:
  mlir::LLVMTypeConverter type_converter;
};

mlir::Type getExceptInfoType(LLVMTypeHelper &type_helper) {
  mlir::Type elems[] = {
      type_helper.ptr(type_helper.i(8)),
      type_helper.i(32),
      type_helper.ptr(type_helper.i(8)),
  };
  return mlir::LLVM::LLVMStructType::getLiteral(&type_helper.get_context(),
                                                elems);
}

mlir::LLVM::LLVMStructType get_array_type(mlir::TypeConverter &converter,
                                          mlir::MemRefType type) {
  assert(type);
  auto ctx = type.getContext();
  auto i8p = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8));
  auto i64 = mlir::IntegerType::get(ctx, 64);
  auto dataType = converter.convertType(type.getElementType());
  assert(dataType);
  if (type.getRank() > 0) {
    auto shapeType = mlir::LLVM::LLVMArrayType::get(
        i64, static_cast<unsigned>(type.getRank()));
    const mlir::Type members[] = {
        i8p,                                        // 0, meminfo
        i8p,                                        // 1, parent
        i64,                                        // 2, nitems
        i64,                                        // 3, itemsize
        mlir::LLVM::LLVMPointerType::get(dataType), // 4, data
        shapeType,                                  // 5, shape
        shapeType,                                  // 6, strides
    };
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, members);
  } else {
    const mlir::Type members[] = {
        i8p,                                        // 0, meminfo
        i8p,                                        // 1, parent
        i64,                                        // 2, nitems
        i64,                                        // 3, itemsize
        mlir::LLVM::LLVMPointerType::get(dataType), // 4, data
    };
    return mlir::LLVM::LLVMStructType::getLiteral(ctx, members);
  }
}

template <typename F> void flatten_type(mlir::Type type, F &&func) {
  if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
    for (auto elem : struct_type.getBody()) {
      flatten_type(elem, std::forward<F>(func));
    }
  } else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
    auto elem = arr_type.getElementType();
    auto size = arr_type.getNumElements();
    for (unsigned i = 0; i < size; ++i) {
      flatten_type(elem, std::forward<F>(func));
    }
  } else {
    func(type);
  }
}

template <typename F>
mlir::Value unflatten(mlir::Type type, mlir::Location loc,
                      mlir::OpBuilder &builder, F &&next_func) {
  namespace mllvm = mlir::LLVM;
  if (auto struct_type = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
    mlir::Value val = builder.create<mllvm::UndefOp>(loc, struct_type);
    for (auto elem : llvm::enumerate(struct_type.getBody())) {
      auto elem_index =
          builder.getI64ArrayAttr(static_cast<int64_t>(elem.index()));
      auto elem_type = elem.value();
      auto elem_val =
          unflatten(elem_type, loc, builder, std::forward<F>(next_func));
      val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val,
                                                      elem_index);
    }
    return val;
  } else if (auto arr_type = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
    auto elem_type = arr_type.getElementType();
    auto size = arr_type.getNumElements();
    mlir::Value val = builder.create<mllvm::UndefOp>(loc, arr_type);
    for (unsigned i = 0; i < size; ++i) {
      auto elem_index = builder.getI64ArrayAttr(static_cast<int64_t>(i));
      auto elem_val =
          unflatten(elem_type, loc, builder, std::forward<F>(next_func));
      val = builder.create<mlir::LLVM::InsertValueOp>(loc, val, elem_val,
                                                      elem_index);
    }
    return val;
  } else {
    return next_func();
  }
}

void write_memref_desc(llvm::raw_ostream &os, mlir::MemRefType memref_type) {
  if (memref_type.hasRank()) {
    auto rank = memref_type.getRank();
    assert(rank >= 0);
    if (rank > 0) {
      os << memref_type.getRank() << "x";
    }
  } else {
    os << "?x";
  }
  memref_type.getElementType().print(os);
}

std::string gen_to_memref_conversion_func_name(mlir::MemRefType memref_type) {
  assert(memref_type);
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << "__convert_to_memref_";
  write_memref_desc(ss, memref_type);
  ss.flush();
  return ret;
}

std::string gen_from_memref_conversion_func_name(mlir::MemRefType memref_type) {
  assert(memref_type);
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << "__convert_from_memref_";
  write_memref_desc(ss, memref_type);
  ss.flush();
  return ret;
}

mlir::Value div_strides(mlir::Location loc, mlir::OpBuilder &builder,
                        mlir::Value strides, mlir::Value m) {
  auto array_type = strides.getType().cast<mlir::LLVM::LLVMArrayType>();
  mlir::Value array = builder.create<mlir::LLVM::UndefOp>(loc, array_type);
  auto count = array_type.getNumElements();
  for (unsigned i = 0; i < count; ++i) {
    auto index = builder.getI64ArrayAttr(i);
    auto prev = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, array_type.getElementType(), strides, index);
    auto val = builder.create<mlir::LLVM::SDivOp>(loc, prev, m);
    array = builder.create<mlir::LLVM::InsertValueOp>(loc, array, val, index);
  }
  return array;
}

mlir::Value mul_strides(mlir::Location loc, mlir::OpBuilder &builder,
                        mlir::Value strides, mlir::Value m) {
  auto array_type = strides.getType().cast<mlir::LLVM::LLVMArrayType>();
  mlir::Value array = builder.create<mlir::LLVM::UndefOp>(loc, array_type);
  auto count = array_type.getNumElements();
  for (unsigned i = 0; i < count; ++i) {
    auto index = builder.getI64ArrayAttr(i);
    auto prev = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, array_type.getElementType(), strides, index);
    auto val = builder.create<mlir::LLVM::MulOp>(loc, prev, m);
    array = builder.create<mlir::LLVM::InsertValueOp>(loc, array, val, index);
  }
  return array;
}

unsigned item_size(mlir::Type type) {
  if (auto inttype = type.dyn_cast<mlir::IntegerType>()) {
    assert((inttype.getWidth() % 8) == 0);
    return inttype.getWidth() / 8;
  }
  if (auto floattype = type.dyn_cast<mlir::FloatType>()) {
    assert((floattype.getWidth() % 8) == 0);
    return floattype.getWidth() / 8;
  }
  llvm_unreachable("item_size: invalid type");
}

mlir::FuncOp
get_to_memref_conversion_func(mlir::ModuleOp module, mlir::OpBuilder &builder,
                              mlir::MemRefType memrefType,
                              mlir::LLVM::LLVMStructType src_type,
                              mlir::LLVM::LLVMStructType dst_type) {
  assert(memrefType);
  assert(src_type);
  assert(dst_type);
  auto func_name = gen_to_memref_conversion_func_name(memrefType);
  if (auto func = module.lookupSymbol<mlir::FuncOp>(func_name)) {
    assert(func.getType().getNumResults() == 1);
    assert(func.getType().getResult(0) == dst_type);
    return func;
  }
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), src_type, dst_type);
  auto loc = builder.getUnknownLoc();
  auto new_func = plier::add_function(builder, module, func_name, func_type);
  auto alwaysinline =
      mlir::StringAttr::get(builder.getContext(), "alwaysinline");
  new_func->setAttr("passthrough",
                    mlir::ArrayAttr::get(builder.getContext(), alwaysinline));
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto block = new_func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  namespace mllvm = mlir::LLVM;
  mlir::Value arg = block->getArgument(0);
  auto extract = [&](unsigned index) {
    auto res_type = src_type.getBody()[index];
    auto i = builder.getI64ArrayAttr(index);
    return builder.create<mllvm::ExtractValueOp>(loc, res_type, arg, i);
  };
  auto meminfo = extract(0);
  auto ptr = extract(4);
  auto rank = memrefType.getRank();
  auto shape = (rank > 0 ? extract(5) : mlir::Value());
  auto strides = (rank > 0 ? extract(6) : mlir::Value());
  auto i64 = mlir::IntegerType::get(builder.getContext(), 64);
  auto offset =
      builder.create<mllvm::ConstantOp>(loc, i64, builder.getI64IntegerAttr(0));
  mlir::Value res = builder.create<mllvm::UndefOp>(loc, dst_type);
  auto meminfo_casted =
      builder.create<mllvm::BitcastOp>(loc, ptr.getType(), meminfo);
  auto itemsize = builder.create<mllvm::ConstantOp>(
      loc, i64,
      builder.getI64IntegerAttr(item_size(memrefType.getElementType())));
  auto insert = [&](unsigned index, mlir::Value val) {
    auto i = builder.getI64ArrayAttr(index);
    res = builder.create<mllvm::InsertValueOp>(loc, res, val, i);
  };
  insert(0, meminfo_casted);
  insert(1, ptr);
  insert(2, offset);
  if (rank > 0) {
    insert(3, shape);
    insert(4, div_strides(loc, builder, strides, itemsize));
  }
  builder.create<mllvm::ReturnOp>(loc, res);
  return new_func;
}

mlir::FuncOp
get_from_memref_conversion_func(mlir::ModuleOp module, mlir::OpBuilder &builder,
                                mlir::MemRefType memrefType,
                                mlir::LLVM::LLVMStructType src_type,
                                mlir::LLVM::LLVMStructType dst_type) {
  assert(memrefType);
  assert(src_type);
  assert(dst_type);
  auto func_name = gen_from_memref_conversion_func_name(memrefType);
  if (auto func = module.lookupSymbol<mlir::FuncOp>(func_name)) {
    assert(func.getType().getNumResults() == 1);
    assert(func.getType().getResult(0) == dst_type);
    return func;
  }
  auto func_type =
      mlir::FunctionType::get(builder.getContext(), src_type, dst_type);
  auto loc = builder.getUnknownLoc();
  auto new_func = plier::add_function(builder, module, func_name, func_type);
  auto alwaysinline =
      mlir::StringAttr::get(builder.getContext(), "alwaysinline");
  new_func->setAttr("passthrough",
                    mlir::ArrayAttr::get(builder.getContext(), alwaysinline));
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto block = new_func.addEntryBlock();
  builder.setInsertionPointToStart(block);
  namespace mllvm = mlir::LLVM;
  mlir::Value arg = block->getArgument(0);
  auto i8ptr_type = mllvm::LLVMPointerType::get(builder.getIntegerType(8));
  auto i64_type = builder.getIntegerType(64);
  auto extract = [&](unsigned index) {
    auto res_type = src_type.getBody()[index];
    auto i = builder.getI64ArrayAttr(index);
    return builder.create<mllvm::ExtractValueOp>(loc, res_type, arg, i);
  };
  auto meminfo = builder.create<mllvm::BitcastOp>(loc, i8ptr_type, extract(0));
  auto orig_ptr = extract(1);
  auto offset = extract(2);
  auto rank = memrefType.getRank();
  auto shape = (rank > 0 ? extract(3) : mlir::Value());
  auto strides = (rank > 0 ? extract(4) : mlir::Value());
  auto ptr = builder.create<mllvm::GEPOp>(loc, orig_ptr.getType(), orig_ptr,
                                          offset.getResult());
  mlir::Value res = builder.create<mllvm::UndefOp>(loc, dst_type);
  auto null = builder.create<mllvm::NullOp>(loc, i8ptr_type);
  mlir::Value nitems = builder.create<mllvm::ConstantOp>(
      loc, i64_type, builder.getI64IntegerAttr(1));
  for (int64_t i = 0; i < rank; ++i) {
    auto dim = builder.create<mllvm::ExtractValueOp>(
        loc, nitems.getType(), shape, builder.getI64ArrayAttr(i));
    nitems = builder.create<mllvm::MulOp>(loc, nitems, dim);
  }
  auto itemsize = builder.create<mllvm::ConstantOp>(
      loc, i64_type,
      builder.getI64IntegerAttr(item_size(memrefType.getElementType())));
  auto insert = [&](unsigned index, mlir::Value val) {
    auto i = builder.getI64ArrayAttr(index);
    res = builder.create<mllvm::InsertValueOp>(loc, res, val, i);
  };
  insert(0, meminfo);
  insert(1, null); // parent
  insert(2, nitems);
  insert(3, itemsize);
  insert(4, ptr);
  if (rank > 0) {
    insert(5, shape);
    insert(6, mul_strides(loc, builder, strides, itemsize));
  }
  builder.create<mllvm::ReturnOp>(loc, res);
  return new_func;
}

mlir::Attribute get_fastmath_attrs(mlir::MLIRContext &ctx) {
  auto add_pair = [&](auto name, auto val) {
    const mlir::Attribute attrs[] = {mlir::StringAttr::get(&ctx, name),
                                     mlir::StringAttr::get(&ctx, val)};
    return mlir::ArrayAttr::get(&ctx, attrs);
  };
  const mlir::Attribute attrs[] = {
      add_pair("denormal-fp-math", "preserve-sign,preserve-sign"),
      add_pair("denormal-fp-math-f32", "ieee,ieee"),
      add_pair("no-infs-fp-math", "true"),
      add_pair("no-nans-fp-math", "true"),
      add_pair("no-signed-zeros-fp-math", "true"),
      add_pair("unsafe-fp-math", "true"),
      add_pair(plier::attributes::getFastmathName(), "1"),
  };
  return mlir::ArrayAttr::get(&ctx, attrs);
}

mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value src, mlir::Type dstType) {
  auto srcType = src.getType();
  if (srcType == dstType) {
    return src;
  }

  return builder.create<mlir::LLVM::DialectCastOp>(loc, dstType, src);
}

mlir::LogicalResult fix_func_sig(LLVMTypeHelper &type_helper,
                                 mlir::FuncOp func) {
  if (func.isPrivate()) {
    return mlir::success();
  }
  if (func->getAttr(plier::attributes::getFastmathName())) {
    func->setAttr("passthrough", get_fastmath_attrs(*func.getContext()));
  }
  auto old_type = func.getType();
  assert(old_type.getNumResults() <= 1);
  auto &ctx = *old_type.getContext();
  llvm::SmallVector<mlir::Type> args;

  auto ptr = [&](auto arg) { return type_helper.ptr(arg); };

  unsigned index = 0;
  auto add_arg = [&](mlir::Type type) {
    args.push_back(type);
    auto ret = func.getBody().insertArgument(index, type);
    ++index;
    return ret;
  };

  auto get_res_type = [&](mlir::Type type) -> mlir::Type {
    if (auto memreftype = type.dyn_cast<mlir::MemRefType>()) {
      return get_array_type(type_helper.get_type_converter(), memreftype);
    }
    return type;
  };

  auto orig_ret_type =
      (old_type.getNumResults() != 0 ? get_res_type(old_type.getResult(0))
                                     : type_helper.ptr(type_helper.i(8)));

  if (!type_helper.get_type_converter().convertType(orig_ret_type)) {
    return mlir::failure();
  }

  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(&func.getBody().front());

  auto loc = builder.getUnknownLoc();
  llvm::SmallVector<mlir::Value> new_args;
  auto process_arg = [&](mlir::Type type) {
    if (auto memrefType = type.dyn_cast<mlir::MemRefType>()) {
      new_args.clear();
      auto arr_type =
          get_array_type(type_helper.get_type_converter(), memrefType);
      flatten_type(arr_type, [&](mlir::Type new_type) {
        new_args.push_back(add_arg(new_type));
      });
      auto it = new_args.begin();
      mlir::Value desc = unflatten(arr_type, loc, builder, [&]() {
        auto ret = *it;
        ++it;
        return ret;
      });

      auto mod = mlir::cast<mlir::ModuleOp>(func->getParentOp());
      auto dst_type = type_helper.get_type_converter().convertType(memrefType);
      assert(dst_type);
      auto conv_func = get_to_memref_conversion_func(
          mod, builder, memrefType, arr_type,
          dst_type.cast<mlir::LLVM::LLVMStructType>());
      auto converted =
          builder.create<mlir::CallOp>(loc, conv_func, desc).getResult(0);
      auto casted = builder.createOrFold<plier::CastOp>(
          loc, memrefType, converted); // TODO: replace castop with dialect cast
      func.getBody().getArgument(index).replaceAllUsesWith(casted);
      func.getBody().eraseArgument(index);
    } else {
      args.push_back(type);
      ++index;
    }
  };

  add_arg(ptr(orig_ret_type));
  add_arg(ptr(ptr(getExceptInfoType(type_helper))));

  auto old_args = old_type.getInputs();
  for (auto arg : old_args) {
    process_arg(arg);
  }
  auto ret_type = mlir::IntegerType::get(&ctx, 32);
  func.setType(mlir::FunctionType::get(&ctx, args, ret_type));
  return mlir::success();
}

struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::ReturnOp> {
  ReturnOpLowering(mlir::MLIRContext *ctx, mlir::TypeConverter &converter)
      : OpRewritePattern(ctx), type_converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = op->getParentOfType<mlir::FuncOp>();
    if (nullptr == parent || parent.isPrivate()) {
      return mlir::failure();
    }

    auto insert_ret = [&]() {
      auto ctx = op.getContext();
      auto ret_type = mlir::IntegerType::get(ctx, 32);
      auto ll_ret_type = mlir::IntegerType::get(ctx, 32);
      mlir::Value ret = rewriter.create<mlir::LLVM::ConstantOp>(
          op.getLoc(), ll_ret_type, mlir::IntegerAttr::get(ret_type, 0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op, ret);
    };

    auto loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto addr = op->getParentRegion()->front().getArgument(0);
    if (op.getNumOperands() == 0) {
      assert(addr.getType().isa<mlir::LLVM::LLVMPointerType>());
      auto null_type =
          addr.getType().cast<mlir::LLVM::LLVMPointerType>().getElementType();
      auto ll_val = rewriter.create<mlir::LLVM::NullOp>(op.getLoc(), null_type);
      rewriter.create<mlir::LLVM::StoreOp>(loc, ll_val, addr);
      insert_ret();
      return mlir::success();
    } else if (op.getNumOperands() == 1) {
      mlir::Value val = op.getOperand(0);
      auto orig_type = val.getType();
      auto llRetType = type_converter.convertType(orig_type);
      if (!llRetType) {
        return mlir::failure();
      }
      val = rewriter.createOrFold<plier::CastOp>(
          loc, llRetType, val); // TODO: replace castop with dialect cast
      if (auto memref_type = orig_type.dyn_cast<mlir::MemRefType>()) {
        auto dst_type = get_array_type(type_converter, memref_type)
                            .cast<mlir::LLVM::LLVMStructType>();
        auto mod = op->getParentOfType<mlir::ModuleOp>();
        auto func = get_from_memref_conversion_func(
            mod, rewriter, memref_type,
            llRetType.cast<mlir::LLVM::LLVMStructType>(), dst_type);
        val = rewriter.create<mlir::CallOp>(loc, func, val).getResult(0);
      }
      rewriter.create<mlir::LLVM::StoreOp>(loc, val, addr);
      insert_ret();
      return mlir::success();
    } else {
      return mlir::failure();
    }
  }

private:
  mlir::TypeConverter &type_converter;
};

// Remove redundant bitcasts we have created on PreLowering
struct RemoveBitcasts : public mlir::OpRewritePattern<mlir::LLVM::BitcastOp> {
  using mlir::OpRewritePattern<mlir::LLVM::BitcastOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::BitcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getType() == op.getOperand().getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return mlir::success();
    }
    return mlir::failure();
  }
};

template <typename Op>
struct ApplyFastmathFlags : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto parent = mlir::cast<mlir::LLVM::LLVMFuncOp>(op->getParentOp());
    bool changed = false;

    rewriter.startRootUpdate(op);
    auto fmf = op.fastmathFlags();
    getFastmathFlags(parent, [&](auto flag) {
      if (!mlir::LLVM::bitEnumContains(fmf, flag)) {
        fmf = fmf | flag;
        changed = true;
      }
    });
    if (changed) {
      op.fastmathFlagsAttr(mlir::LLVM::FMFAttr::get(op.getContext(), fmf));
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.cancelRootUpdate(op);
    }

    return mlir::success(changed);
  }

private:
  template <typename F>
  static void getFastmathFlags(mlir::LLVM::LLVMFuncOp func, F &&sink) {
    if (func->hasAttr(plier::attributes::getFastmathName())) {
      sink(mlir::LLVM::FastmathFlags::fast);
    }
  }
};

struct AllocOpLowering : public mlir::AllocLikeOpLLVMLowering {
  AllocOpLowering(mlir::LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(mlir::memref::AllocOp::getOperationName(),
                                converter) {}

  std::tuple<mlir::Value, mlir::Value>
  allocateBuffer(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 mlir::Value sizeBytes, mlir::Operation *op) const override {
    auto allocOp = mlir::cast<mlir::memref::AllocOp>(op);
    auto memRefType = allocOp.getType();
    mlir::Value alignment;
    if (auto alignmentAttr = allocOp.alignment()) {
      alignment = createIndexConstant(rewriter, loc, *alignmentAttr);
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      // In the case where no alignment is specified, we may want to override
      // `malloc's` behavior. `malloc` typically aligns at the size of the
      // biggest scalar on a target HW. For non-scalars, use the natural
      // alignment of the LLVM type given by the LLVM DataLayout.
      alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
    } else {
      alignment = createIndexConstant(
          rewriter, loc, 32 /*item_size(memRefType.getElementType())*/);
    }
    alignment = rewriter.create<mlir::LLVM::TruncOp>(
        loc, rewriter.getIntegerType(32), alignment);

    auto mod = allocOp->getParentOfType<mlir::ModuleOp>();
    auto meminfo_ptr =
        createAllocCall(loc, "NRT_MemInfo_alloc_safe_aligned", getVoidPtrType(),
                        {sizeBytes, alignment}, mod, rewriter);
    auto data_ptr =
        createAllocCall(loc, "NRT_MemInfo_data_fast", getVoidPtrType(),
                        {meminfo_ptr}, mod, rewriter);

    auto elem_ptr_type =
        mlir::LLVM::LLVMPointerType::get(memRefType.getElementType());
    auto bitcast = [&](mlir::Value val) {
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, elem_ptr_type, val);
    };

    return std::make_tuple(bitcast(meminfo_ptr), bitcast(data_ptr));
  }

private:
  mlir::Value createAllocCall(mlir::Location loc, mlir::StringRef name,
                              mlir::Type ptrType,
                              mlir::ArrayRef<mlir::Value> params,
                              mlir::ModuleOp module,
                              mlir::ConversionPatternRewriter &rewriter) const {
    using namespace mlir;
    SmallVector<Type, 2> paramTypes;
    auto allocFuncOp = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    if (!allocFuncOp) {
      for (Value param : params)
        paramTypes.push_back(param.getType());
      auto allocFuncType =
          LLVM::LLVMFunctionType::get(getVoidPtrType(), paramTypes);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      allocFuncOp = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                      name, allocFuncType);
    }
    auto allocFuncSymbol = rewriter.getSymbolRefAttr(allocFuncOp);
    auto allocatedPtr = rewriter
                            .create<LLVM::CallOp>(loc, getVoidPtrType(),
                                                  allocFuncSymbol, params)
                            .getResult(0);
    return rewriter.create<LLVM::BitcastOp>(loc, ptrType, allocatedPtr);
  }
};

struct DeallocOpLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::DeallocOp> {
  using ConvertOpToLLVMPattern<mlir::memref::DeallocOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DeallocOp op,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1 && "dealloc takes one operand");
    mlir::memref::DeallocOp::Adaptor transformed(operands);

    // Insert the `free` declaration if it is not already present.
    auto freeFunc = op->getParentOfType<mlir::ModuleOp>()
                        .lookupSymbol<mlir::LLVM::LLVMFuncOp>("NRT_decref");
    if (!freeFunc) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(
          op->getParentOfType<mlir::ModuleOp>().getBody());
      freeFunc = rewriter.create<mlir::LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), "NRT_decref",
          mlir::LLVM::LLVMFunctionType::get(getVoidType(), getVoidPtrType()));
    }

    mlir::MemRefDescriptor memref(transformed.memref());
    mlir::Value casted = rewriter.create<mlir::LLVM::BitcastOp>(
        op.getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op.getLoc()));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        op, mlir::TypeRange(), rewriter.getSymbolRefAttr(freeFunc), casted);
    return mlir::success();
  }
};

struct ReshapeLowering
    : public mlir::ConvertOpToLLVMPattern<mlir::memref::ReshapeOp> {
  using ConvertOpToLLVMPattern<mlir::memref::ReshapeOp>::ConvertOpToLLVMPattern;

  explicit ReshapeLowering(mlir::LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<mlir::memref::ReshapeOp>(converter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReshapeOp op,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 2);
    mlir::memref::ReshapeOp::Adaptor transformed(operands);

    auto converter = getTypeConverter();
    auto dstType = converter->convertType(op.getType());
    if (!dstType) {
      return mlir::failure();
    }

    mlir::MemRefDescriptor source(transformed.source());
    mlir::MemRefDescriptor shape(transformed.shape());

    auto loc = op.getLoc();
    auto result = mlir::MemRefDescriptor::undef(rewriter, loc, dstType);
    result.setAllocatedPtr(rewriter, loc, source.allocatedPtr(rewriter, loc));
    result.setAlignedPtr(rewriter, loc, source.alignedPtr(rewriter, loc));
    result.setOffset(rewriter, loc, source.offset(rewriter, loc));

    auto memRefType = op.getType().cast<mlir::MemRefType>();
    auto numDims = memRefType.getRank();
    llvm::SmallVector<mlir::Value> sizes(static_cast<unsigned>(numDims));
    auto indexType = getIndexType();
    for (unsigned i = 0; i < numDims; ++i) {
      auto ind = createIndexConstant(rewriter, loc, i);
      mlir::Value dataPtr =
          getStridedElementPtr(loc, memRefType, shape, ind, rewriter);
      auto size = rewriter.create<mlir::LLVM::LoadOp>(loc, dataPtr).getResult();
      if (size.getType() != indexType) {
        size = rewriter.create<mlir::LLVM::ZExtOp>(loc, indexType, size);
      }
      result.setSize(rewriter, loc, i, size);
      sizes[i] = size;
    }

    // Strides: iterate sizes in reverse order and multiply.
    int64_t stride = 1;
    mlir::Value runningStride = createIndexConstant(rewriter, loc, 1);
    for (auto i = static_cast<unsigned>(memRefType.getRank()); i-- > 0;) {
      result.setStride(rewriter, loc, i, runningStride);

      int64_t size = memRefType.getShape()[i];
      if (size == 0)
        continue;
      bool useSizeAsStride = stride == 1;
      if (size == mlir::ShapedType::kDynamicSize)
        stride = mlir::ShapedType::kDynamicSize;
      if (stride != mlir::ShapedType::kDynamicSize)
        stride *= size;

      if (useSizeAsStride)
        runningStride = sizes[i];
      else if (stride == mlir::ShapedType::kDynamicSize)
        runningStride =
            rewriter.create<mlir::LLVM::MulOp>(loc, runningStride, sizes[i]);
      else
        runningStride =
            createIndexConstant(rewriter, loc, static_cast<uint64_t>(stride));
    }

    rewriter.replaceOp(op, static_cast<mlir::Value>(result));
    return mlir::success();
  }
};

class LLVMFunctionPass : public mlir::OperationPass<mlir::LLVM::LLVMFuncOp> {
public:
  using OperationPass<mlir::LLVM::LLVMFuncOp>::OperationPass;

  /// The polymorphic API that runs the pass over the currently held function.
  virtual void runOnFunction() = 0;

  /// The polymorphic API that runs the pass over the currently held operation.
  void runOnOperation() final {
    if (!getFunction().isExternal())
      runOnFunction();
  }

  /// Return the current function being transformed.
  mlir::LLVM::LLVMFuncOp getFunction() { return this->getOperation(); }
};

void copyAttrs(mlir::Operation *src, mlir::Operation *dst) {
  const mlir::StringRef attrs[] = {
      plier::attributes::getFastmathName(),
      plier::attributes::getParallelName(),
      plier::attributes::getMaxConcurrencyName(),
  };
  for (auto name : attrs) {
    if (auto attr = src->getAttr(name)) {
      dst->setAttr(name, attr);
    }
  }
}

struct LowerParallel : public mlir::OpRewritePattern<plier::ParallelOp> {
  LowerParallel(mlir::MLIRContext *context)
      : OpRewritePattern(context), converter(context) {}

  mlir::LogicalResult
  matchAndRewrite(plier::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto num_loops = op.getNumLoops();
    llvm::SmallVector<mlir::Value> context_vars;
    llvm::SmallVector<mlir::Operation *> context_constants;
    llvm::DenseSet<mlir::Value> context_vars_set;
    auto add_context_var = [&](mlir::Value value) {
      if (0 != context_vars_set.count(value)) {
        return;
      }
      context_vars_set.insert(value);
      if (auto op = value.getDefiningOp()) {
        mlir::ConstantOp a;
        if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
          context_constants.emplace_back(op);
          return;
        }
      }
      context_vars.emplace_back(value);
    };

    auto is_defined_inside = [&](mlir::Value value) {
      auto &this_region = op.getLoopBody();
      auto op_region = value.getParentRegion();
      assert(nullptr != op_region);
      do {
        if (op_region == &this_region) {
          return true;
        }
        op_region = op_region->getParentRegion();
      } while (nullptr != op_region);
      return false;
    };

    if (op->walk([&](mlir::Operation *inner) -> mlir::WalkResult {
            if (op != inner) {
              for (auto arg : inner->getOperands()) {
                if (!is_defined_inside(arg)) {
                  add_context_var(arg);
                }
              }
            }
            return mlir::WalkResult::advance();
          }).wasInterrupted()) {
      return mlir::failure();
    }

    auto context_type = [&]() -> mlir::LLVM::LLVMStructType {
      llvm::SmallVector<mlir::Type> fields;
      fields.reserve(context_vars.size());
      for (auto var : context_vars) {
        auto type = converter.convertType(var.getType());
        if (!type) {
          return {};
        }
        fields.emplace_back(type);
      }
      return mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), fields);
    }();

    if (!context_type) {
      return mlir::failure();
    }

    plier::AllocaInsertionPoint allocaInsertionPoint(op);

    auto context_ptr_type = mlir::LLVM::LLVMPointerType::get(context_type);

    auto loc = op.getLoc();
    auto indexType = rewriter.getIndexType();
    auto llvmIndexType = mlir::IntegerType::get(op.getContext(), 64); // TODO
    auto to_llvm_index = [&](mlir::Value val) -> mlir::Value {
      if (val.getType() != llvmIndexType) {
        return rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmIndexType, val);
      }
      return val;
    };
    auto from_llvm_index = [&](mlir::Value val) -> mlir::Value {
      if (val.getType() != indexType) {
        return doCast(rewriter, loc, val, indexType);
      }
      return val;
    };
    auto llvm_i32_type = mlir::IntegerType::get(op.getContext(), 32);
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvm_i32_type, rewriter.getI32IntegerAttr(0));
    auto context = allocaInsertionPoint.insert(rewriter, [&]() {
      auto one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvm_i32_type, rewriter.getI32IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, context_ptr_type, one,
                                                   0);
    });

    for (auto it : llvm::enumerate(context_vars)) {
      auto type = context_type.getBody()[it.index()];
      auto llvmVal = rewriter.create<plier::CastOp>(
          loc, type, it.value()); // TODO: replace castop with dialect cast
      auto i = rewriter.getI32IntegerAttr(static_cast<int32_t>(it.index()));
      mlir::Value indices[] = {
          zero, rewriter.create<mlir::LLVM::ConstantOp>(loc, llvm_i32_type, i)};
      auto pointer_type = mlir::LLVM::LLVMPointerType::get(type);
      auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, pointer_type, context,
                                                    indices);
      rewriter.create<mlir::LLVM::StoreOp>(loc, llvmVal, ptr);
    }
    auto void_ptr_type = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(op.getContext(), 8));
    auto context_abstract =
        rewriter.create<mlir::LLVM::BitcastOp>(loc, void_ptr_type, context);

    auto input_range_type = [&]() {
      const mlir::Type members[] = {
          llvmIndexType, // lower_bound
          llvmIndexType, // upper_bound
          llvmIndexType, // step
      };
      return mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), members);
    }();
    auto input_range_ptr = mlir::LLVM::LLVMPointerType::get(input_range_type);
    auto range_type = [&]() {
      const mlir::Type members[] = {
          llvmIndexType, // lower_bound
          llvmIndexType, // upper_bound
      };
      return mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), members);
    }();
    auto range_ptr = mlir::LLVM::LLVMPointerType::get(range_type);
    auto func_type = [&]() {
      const mlir::Type args[] = {
          range_ptr,    // bounds
          indexType,    // thread index
          void_ptr_type // context
      };
      return mlir::FunctionType::get(op.getContext(), args, {});
    }();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    auto outlined_func = [&]() -> mlir::FuncOp {
      auto func = [&]() {
        auto parent_func = op->getParentOfType<mlir::FuncOp>();
        assert(parent_func);
        auto func_name = [&]() {
          auto old_name = parent_func.getName();
          for (int i = 0;; ++i) {
            auto name =
                (0 == i
                     ? (llvm::Twine(old_name) + "_outlined").str()
                     : (llvm::Twine(old_name) + "_outlined_" + llvm::Twine(i))
                           .str());
            if (!mod.lookupSymbol<mlir::FuncOp>(name)) {
              return name;
            }
          }
        }();

        auto func = plier::add_function(rewriter, mod, func_name, func_type);
        copyAttrs(parent_func, func);
        return func;
      }();
      mlir::BlockAndValueMapping mapping;
      auto &old_entry = op.getLoopBody().front();
      auto entry = func.addEntryBlock();
      auto loc = rewriter.getUnknownLoc();
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(entry);
      auto pos0 = rewriter.getI64ArrayAttr(0);
      auto pos1 = rewriter.getI64ArrayAttr(1);
      for (unsigned i = 0; i < num_loops; ++i) {
        auto arg = entry->getArgument(0);
        const mlir::Value indices[] = {rewriter.create<mlir::LLVM::ConstantOp>(
            loc, llvm_i32_type,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(i)))};
        auto ptr =
            rewriter.create<mlir::LLVM::GEPOp>(loc, range_ptr, arg, indices);
        auto dims = rewriter.create<mlir::LLVM::LoadOp>(loc, ptr);
        auto lower = rewriter.create<mlir::LLVM::ExtractValueOp>(
            loc, llvmIndexType, dims, pos0);
        auto upper = rewriter.create<mlir::LLVM::ExtractValueOp>(
            loc, llvmIndexType, dims, pos1);
        mapping.map(old_entry.getArgument(i), from_llvm_index(lower));
        mapping.map(old_entry.getArgument(i + num_loops),
                    from_llvm_index(upper));
      }
      mapping.map(old_entry.getArgument(2 * num_loops),
                  entry->getArgument(1)); // thread index
      for (auto arg : context_constants) {
        rewriter.clone(*arg, mapping);
      }
      auto context_ptr = rewriter.create<mlir::LLVM::BitcastOp>(
          loc, context_ptr_type, entry->getArgument(2));
      auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvm_i32_type, rewriter.getI32IntegerAttr(0));
      for (auto it : llvm::enumerate(context_vars)) {
        auto index = it.index();
        auto oldVal = it.value();
        const mlir::Value indices[] = {
            zero, rewriter.create<mlir::LLVM::ConstantOp>(
                      loc, llvm_i32_type,
                      rewriter.getI32IntegerAttr(static_cast<int32_t>(index)))};
        auto pointer_type =
            mlir::LLVM::LLVMPointerType::get(context_type.getBody()[index]);
        auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, pointer_type,
                                                      context_ptr, indices);
        auto llvmVal = rewriter.create<mlir::LLVM::LoadOp>(loc, ptr);
        auto val = doCast(rewriter, loc, llvmVal, oldVal.getType());
        mapping.map(oldVal, val);
      }
      op.getLoopBody().cloneInto(&func.getBody(), mapping);
      auto &orig_entry = *std::next(func.getBody().begin());
      rewriter.create<mlir::BranchOp>(loc, &orig_entry);
      for (auto &block : func.getBody()) {
        if (auto term = mlir::dyn_cast<plier::YieldOp>(block.getTerminator())) {
          rewriter.eraseOp(term);
          rewriter.setInsertionPointToEnd(&block);
          rewriter.create<mlir::ReturnOp>(loc);
        }
      }
      return func;
    }();

    auto parallel_for = [&]() {
      auto func_name = "dpcomp_parallel_for";
      if (auto sym = mod.lookupSymbol<mlir::FuncOp>(func_name)) {
        return sym;
      }
      const mlir::Type args[] = {
          input_range_ptr, // bounds
          indexType,       // num_loops
          func_type,       // func
          void_ptr_type    // context
      };
      auto parallel_func_type =
          mlir::FunctionType::get(op.getContext(), args, {});
      return plier::add_function(rewriter, mod, func_name, parallel_func_type);
    }();
    auto func_addr = rewriter.create<mlir::ConstantOp>(
        loc, func_type, rewriter.getSymbolRefAttr(outlined_func));

    auto input_ranges = allocaInsertionPoint.insert(rewriter, [&]() {
      auto num_loops_attr = rewriter.getIntegerAttr(llvmIndexType, num_loops);
      auto num_loops_var = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmIndexType, num_loops_attr);
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, input_range_ptr,
                                                   num_loops_var, 0);
    });
    for (unsigned i = 0; i < num_loops; ++i) {
      mlir::Value input_range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, input_range_type);
      auto insert = [&](mlir::Value val, unsigned index) {
        input_range = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, input_range, val, rewriter.getI64ArrayAttr(index));
      };
      insert(to_llvm_index(op.lowerBounds()[i]), 0);
      insert(to_llvm_index(op.upperBounds()[i]), 1);
      insert(to_llvm_index(op.steps()[i]), 2);
      const mlir::Value indices[] = {rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvm_i32_type, rewriter.getI32IntegerAttr(static_cast<int>(i)))};
      auto ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, input_range_ptr,
                                                    input_ranges, indices);
      rewriter.create<mlir::LLVM::StoreOp>(loc, input_range, ptr);
    }

    auto num_loops_var = rewriter.create<mlir::ConstantIndexOp>(loc, num_loops);
    const mlir::Value pf_args[] = {input_ranges, num_loops_var, func_addr,
                                   context_abstract};
    rewriter.create<mlir::CallOp>(loc, parallel_for, pf_args);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mutable mlir::LLVMTypeConverter converter; // TODO
};

struct LowerParallelToCFGPass
    : public mlir::PassWrapper<LowerParallelToCFGPass,
                               mlir::OperationPass<void>> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override final {
    auto &context = getContext();
    mlir::OwningRewritePatternList patterns(&context);
    patterns.insert<LowerParallel>(&getContext());

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct PreLLVMLowering
    : public mlir::PassWrapper<PreLLVMLowering, mlir::FunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnFunction() override final {
    auto &context = getContext();
    LLVMTypeHelper type_helper(context);

    mlir::OwningRewritePatternList patterns(&context);
    auto func = getFunction();
    if (mlir::failed(fix_func_sig(type_helper, func))) {
      signalPassFailure();
      return;
    }

    patterns.insert<ReturnOpLowering>(&context,
                                      type_helper.get_type_converter());

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct PostLLVMLowering
    : public mlir::PassWrapper<PostLLVMLowering, LLVMFunctionPass> {
  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnFunction() override final {
    auto &context = getContext();
    mlir::OwningRewritePatternList patterns(&context);

    patterns.insert<RemoveBitcasts, ApplyFastmathFlags<mlir::LLVM::FAddOp>,
                    ApplyFastmathFlags<mlir::LLVM::FSubOp>,
                    ApplyFastmathFlags<mlir::LLVM::FMulOp>,
                    ApplyFastmathFlags<mlir::LLVM::FDivOp>,
                    ApplyFastmathFlags<mlir::LLVM::FRemOp>,
                    ApplyFastmathFlags<mlir::LLVM::FCmpOp>,
                    ApplyFastmathFlags<mlir::LLVM::CallOp>>(&context);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

struct LowerRetainOp : public mlir::ConvertOpToLLVMPattern<plier::RetainOp> {
  using mlir::ConvertOpToLLVMPattern<plier::RetainOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::RetainOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    plier::RetainOp::Adaptor transformed(operands);
    auto arg = transformed.source();
    if (!arg.getType().isa<mlir::LLVM::LLVMStructType>()) {
      return mlir::failure();
    }

    auto llvmVoidPointerType = getVoidPtrType();
    auto incref_func = [&]() {
      auto mod = op->getParentOfType<mlir::ModuleOp>();
      assert(mod);
      auto func = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>("NRT_incref");
      if (!func) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());
        auto llvmVoidType = getVoidType();
        func = rewriter.create<mlir::LLVM::LLVMFuncOp>(
            rewriter.getUnknownLoc(), "NRT_incref",
            mlir::LLVM::LLVMFunctionType::get(llvmVoidType,
                                              llvmVoidPointerType));
      }
      return func;
    }();

    mlir::MemRefDescriptor source(arg);

    auto loc = op.getLoc();
    mlir::Value ptr = source.allocatedPtr(rewriter, loc);
    ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmVoidPointerType, ptr);
    rewriter.create<mlir::LLVM::CallOp>(loc, incref_func, ptr);
    rewriter.replaceOp(op, arg);

    return mlir::success();
  }
};

struct LowerReduceRankOp
    : public mlir::ConvertOpToLLVMPattern<plier::ReduceRankOp> {
  using mlir::ConvertOpToLLVMPattern<
      plier::ReduceRankOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ReduceRankOp op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    plier::ReduceRankOp::Adaptor transformed(operands);
    auto arg = transformed.source();
    if (!arg.getType().isa<mlir::LLVM::LLVMStructType>())
      return mlir::failure();

    auto dstType = getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::MemRefDescriptor src(arg);
    mlir::MemRefDescriptor dst =
        mlir::MemRefDescriptor::undef(rewriter, loc, dstType);

    dst.setAllocatedPtr(rewriter, loc, dst.allocatedPtr(rewriter, loc));
    dst.setAlignedPtr(rewriter, loc, dst.alignedPtr(rewriter, loc));
    dst.setOffset(rewriter, loc, dst.offset(rewriter, loc));

    auto mapping = op.mapping();
    for (auto it : llvm::enumerate(mapping)) {
      auto index = static_cast<unsigned>(it.index());
      auto originalIndex = static_cast<unsigned>(
          it.value().cast<mlir::IntegerAttr>().getValue().getSExtValue());
      dst.setSize(rewriter, loc, index, src.size(rewriter, loc, originalIndex));
      dst.setStride(rewriter, loc, index,
                    src.stride(rewriter, loc, originalIndex));
    }
    rewriter.replaceOp(op, static_cast<mlir::Value>(dst));
    return mlir::success();
  }
};

struct LowerBuildTuple
    : public mlir::ConvertOpToLLVMPattern<plier::BuildTupleOp> {
  using mlir::ConvertOpToLLVMPattern<
      plier::BuildTupleOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildTupleOp op,
                  llvm::ArrayRef<mlir::Value> /*operands*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto type = converter->convertType(op.getType());
    if (!type) {
      return mlir::failure();
    }
    for (auto arg : op.args()) {
      if (!converter->convertType(arg.getType())) {
        return mlir::failure();
      }
    }

    auto loc = op.getLoc();
    mlir::Value init = rewriter.create<mlir::LLVM::UndefOp>(loc, type);
    for (auto it : llvm::enumerate(op.args())) {
      auto arg = it.value();
      auto newType = converter->convertType(arg.getType());
      assert(newType);
      auto casted = doCast(rewriter, loc, arg, newType);
      auto index = rewriter.getI64ArrayAttr(static_cast<int64_t>(it.index()));
      init =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, init, casted, index);
    }

    rewriter.replaceOp(op, init);
    return mlir::success();
  }
};

struct LowerUndef : public mlir::ConvertOpToLLVMPattern<plier::UndefOp> {
  using mlir::ConvertOpToLLVMPattern<plier::UndefOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UndefOp op, llvm::ArrayRef<mlir::Value> /*operands*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto type = converter->convertType(op.getType());
    if (!type) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, type);
    return mlir::success();
  }
};

template <typename Op>
struct LowerCasts : public mlir::ConvertOpToLLVMPattern<Op> {
  using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    typename Op::Adaptor transformed(operands);
    rewriter.replaceOpWithNewOp<mlir::LLVM::DialectCastOp>(op, op.getType(),
                                                           transformed.value());
    return mlir::success();
  }
};

// Copypasted from mlir
struct LLVMLoweringPass
    : public mlir::PassWrapper<LLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    using namespace mlir;
    auto &context = getContext();
    auto options = getLLVMOptions(context);
    if (options.useBarePtrCallConv && options.emitCWrappers) {
      getOperation().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            options.dataLayout.getStringRepresentation(),
            [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    ModuleOp m = getOperation();

    LLVMTypeConverter typeConverter(&context, options);
    populateToLLVMAdditionalTypeConversion(typeConverter);
    OwningRewritePatternList patterns(&context);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateLinalgToLLVMConversionPatterns(typeConverter, patterns);

    patterns.insert<
        // clang-format off
        LowerUndef,
        LowerCasts<plier::CastOp>,
        LowerCasts<plier::SignCastOp>,
        LowerBuildTuple,
        LowerRetainOp,
        AllocOpLowering,
        DeallocOpLowering,
        ReshapeLowering,
        LowerReduceRankOp
        // clang-format on
        >(typeConverter);

    LLVMConversionTarget target(context);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(m.getContext(),
                               options.dataLayout.getStringRepresentation()));
  }

private:
};

void populate_lower_to_llvm_pipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<LowerParallelToCFGPass>());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::FuncOp>(std::make_unique<PreLLVMLowering>());
  pm.addPass(std::make_unique<LLVMLoweringPass>());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
      std::make_unique<PostLLVMLowering>());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void register_lower_to_llvm_pipeline(plier::PipelineRegistry &registry) {
  registry.register_pipeline([](auto sink) {
    auto stage = get_lower_lowering_stage();
    sink(lower_to_llvm_pipeline_name(), {stage.begin}, {stage.end}, {},
         &populate_lower_to_llvm_pipeline);
  });
}

llvm::StringRef lower_to_llvm_pipeline_name() { return "lower_to_llvm"; }
