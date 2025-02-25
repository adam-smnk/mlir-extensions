//===- XeGPUDialect.cpp - XeGPU dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeGPU dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/XeGPU/IR/XeGPUOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeUtilities.h>

#include <numeric>

#include "imex/Utils/DebugUtils.h"

namespace imex {
namespace xegpu {

void XeGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/XeGPU/IR/XeGPUOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/XeGPU/IR/XeGPUOps.cpp.inc>
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <imex/Dialect/XeGPU/IR/XeGPUOpsAttrs.cpp.inc>
      >();
}

// custom parser for XeGPU_TensorDesc (shape and type parameter)
static mlir::LogicalResult parseShapeAndType(mlir::AsmParser &parser,
                                             llvm::SmallVector<int64_t> &shape,
                                             mlir::Type &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();
  shape = std::move(dimensions);

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();
  type = std::move(t);

  return mlir::success();
}

// custom printer for XeGPU_TensorDesc (shape and type parameter)
static void printShapeAndType(mlir::AsmPrinter &printer,
                              llvm::ArrayRef<int64_t> shape, mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
}

static mlir::LogicalResult parseTensorDescAttr(mlir::AsmParser &parser,
                                               imex::xegpu::MemoryScope &scope,
                                               mlir::Attribute &encoding) {
  // implies no attrbutes
  if (mlir::failed(parser.parseOptionalComma()))
    return mlir::success();

  auto parseElt = [&]() -> mlir::ParseResult {
    llvm::StringRef nameId;

    if (!parser.parseOptionalKeyword(&nameId, {"memory_scope"})) {
      auto loc = parser.getCurrentLocation();
      if (parser.parseEqual())
        return mlir::failure();

      auto attrOptional =
          ::mlir::FieldParser<::imex::xegpu::MemoryScope,
                              ::imex::xegpu::MemoryScope>::parse(parser);
      if (mlir::failed(attrOptional))
        return parser.emitError(
            loc, "Invalid memory scope attribute specification.\n");
      scope = *attrOptional;
      return mlir::success();
    } else {
      auto loc = parser.getCurrentLocation();
      auto attrOptional = ::mlir::FieldParser<::mlir::Attribute>::parse(parser);
      if (mlir::failed(attrOptional))
        return parser.emitError(
            loc, "Failed to parse XeGPU_TensorDesc parameter 'encoding' which "
                 "is to be a `::mlir::Attribute`.\n");
      encoding = *attrOptional;
      return mlir::success();
    }
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();

  return mlir::success();
}

static void printTensorDescAttr(mlir::AsmPrinter &printer,
                                imex::xegpu::MemoryScope scope,
                                mlir::Attribute encoding) {
  if (scope != imex::xegpu::MemoryScope::GLOBAL)
    printer << ", memory_scope = " << scope;
  if (encoding)
    printer << ", " << encoding;
}

template <typename T>
static mlir::LogicalResult parseArrayList(mlir::AsmParser &parser,
                                          llvm::SmallVector<T> &array,
                                          bool parsePrecedenceEqual = false) {
  mlir::FailureOr<llvm::SmallVector<T>> result;
  // Parse literal '='
  if (parsePrecedenceEqual)
    if (parser.parseEqual())
      return mlir::failure();

  // Parse literal '['
  if (parser.parseLSquare())
    return mlir::failure();

  result = mlir::FieldParser<::llvm::SmallVector<T>>::parse(parser);

  if (::mlir::failed(result))
    return mlir::failure();

  // Parse literal ']'
  if (parser.parseRSquare())
    return mlir::failure();

  array = result.value();
  return mlir::success();
}

template <typename T>
static void printArrayElement(mlir::AsmPrinter &printer,
                              llvm::StringRef keyword,
                              llvm::ArrayRef<T> array) {
  printer << keyword;
  printer << ' ' << "=";
  printer << ' ' << "[";
  printer.printStrippedAttrOrType(array);
  printer << "]";
}

static mlir::LogicalResult
parseSubGroupMapAttrElements(mlir::AsmParser &parser,
                             llvm::SmallVector<unsigned> &layout,
                             llvm::SmallVector<unsigned> &data,
                             llvm::SmallVector<unsigned> &mmaBlockSize) {
  auto parseElt = [&]() -> mlir::LogicalResult {
    return mlir::AsmParser::KeywordSwitch<mlir::LogicalResult>(parser)
        .Case("mma_block_size",
              [&](llvm::StringRef, llvm::SMLoc) {
                return parseArrayList(parser, mmaBlockSize, true);
              })
        .Case("wi_layout",
              [&](llvm::StringRef, llvm::SMLoc) {
                return parseArrayList(parser, layout, true);
              })
        .Case("wi_data",
              [&](llvm::StringRef, llvm::SMLoc) {
                return parseArrayList(parser, data, true);
              })
        .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
          parser.emitError(
              parser.getCurrentLocation(),
              "SubGroupMapAttr Parser meet an unexpected keywoard: ")
              << keyword << "\n";
          return mlir::failure();
        });
  };

  if (parser.parseLBrace())
    return mlir::failure();
  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();
  if (parser.parseRBrace())
    return mlir::failure();

  return mlir::success();
}

static void printSubGroupMapAttrElements(
    mlir::AsmPrinter &printer, llvm::ArrayRef<unsigned> layout,
    llvm::ArrayRef<unsigned> data, llvm::ArrayRef<unsigned> mmaBlockSize) {
  printer << "{";
  if (mmaBlockSize.size()) {
    printArrayElement(printer, "mma_block_size", mmaBlockSize);
    printer << "," << ' ';
  }
  printArrayElement(printer, "wi_layout", layout);
  printer << "," << ' ';
  printArrayElement(printer, "wi_data", data);
  printer << "}";
}

static mlir::LogicalResult
parseWorkGroupMapAttrElements(mlir::AsmParser &parser,
                              llvm::SmallVector<unsigned> &layout,
                              llvm::SmallVector<unsigned> &data) {
  auto parseElt = [&]() -> mlir::LogicalResult {
    return mlir::AsmParser::KeywordSwitch<mlir::LogicalResult>(parser)
        .Case("sg_layout",
              [&](llvm::StringRef, llvm::SMLoc) {
                return parseArrayList(parser, layout, true);
              })
        .Case("sg_data",
              [&](llvm::StringRef, llvm::SMLoc) {
                return parseArrayList(parser, data, true);
              })
        .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
          parser.emitError(
              parser.getCurrentLocation(),
              "WorkGroupMapAttr Parser meet an unexpected keywoard: ")
              << keyword << "\n";
          return mlir::failure();
        });
  };

  if (parser.parseLBrace())
    return mlir::failure();
  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();
  if (parser.parseRBrace())
    return mlir::failure();
  return mlir::success();
}

static void printWorkGroupMapAttrElements(mlir::AsmPrinter &printer,
                                          llvm::ArrayRef<unsigned> layout,
                                          llvm::ArrayRef<unsigned> data) {
  printer << "{";
  printArrayElement(printer, "sg_layout", layout);
  printer << "," << ' ';
  printArrayElement(printer, "sg_data", data);
  printer << "}";
}

mlir::LogicalResult SubGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<unsigned> layout, llvm::ArrayRef<unsigned> data,
    llvm::ArrayRef<unsigned> mmaBlockSize) {

  if (mmaBlockSize.size() != 2 && mmaBlockSize.size() != 0) {
    emitError()
        << "Failed to parse SubGroupMapAttr: mma_block_size should be a "
           "`llvm::ArrayRef<unsigned>` with size 2 or empty. But it got "
        << mmaBlockSize.size() << ".\n";
    return mlir::failure();
  }

  if (layout.size() != 2) {
    emitError() << "Failed to parse SubGroupMapAttr: missing wi_layout which "
                   "is to be a `llvm::ArrayRef<unsigned>` with size 2.\n";
    return mlir::failure();
  }

  if (data.size() != 2) {
    emitError() << "Failed to parse SubGroupMapAttr: missing wi_data which is "
                   "to be a `llvm::ArrayRef<unsigned>` with size 2.\n";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult WorkGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<unsigned> layout, llvm::ArrayRef<unsigned> data) {

  if (layout.size() != 2) {
    emitError() << "Failed to parse WorkGroupMapAttr: missing sg_layout which "
                   "is to be a `llvm::ArrayRef<unsigned>` with size 2.\n";
    return mlir::failure();
  }
  if (data.size() != 2) {
    emitError() << "Failed to parse WorkGroupMapAttr: missing sg_data which is "
                   "to be a `llvm::ArrayRef<unsigned>` with size 2.\n";
    return mlir::failure();
  }
  return mlir::success();
}

mlir::Attribute XeMapAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  imex::xegpu::WorkGroupMapAttr wg;
  imex::xegpu::SubGroupMapAttr sg;
  // Parse literal '<'
  if (parser.parseLess())
    return {};

  auto parseElt = [&]() -> mlir::ParseResult {
    mlir::OptionalParseResult result =
        mlir::AsmParser::KeywordSwitch<mlir::OptionalParseResult>(parser)
            .Case("sg",
                  [&](llvm::StringRef, llvm::SMLoc) {
                    if (parser.parseEqual())
                      return mlir::failure();
                    llvm::SmallVector<unsigned> mmaBlockSize;
                    llvm::SmallVector<unsigned> wiLayout;
                    llvm::SmallVector<unsigned> wiData;
                    if (mlir::failed(parseSubGroupMapAttrElements(
                            parser, wiLayout, wiData, mmaBlockSize)))
                      return mlir::failure();
                    sg = imex::xegpu::SubGroupMapAttr::get(
                        parser.getContext(), wiLayout, wiData, mmaBlockSize);
                    return mlir::success(!!sg);
                  })
            .Case("wg",
                  [&](llvm::StringRef, llvm::SMLoc) {
                    if (parser.parseEqual())
                      return mlir::failure();
                    llvm::SmallVector<unsigned> sgLayout;
                    llvm::SmallVector<unsigned> sgData;
                    if (mlir::failed(parseWorkGroupMapAttrElements(
                            parser, sgLayout, sgData)))
                      return mlir::failure();
                    wg = imex::xegpu::WorkGroupMapAttr::get(parser.getContext(),
                                                            sgLayout, sgData);
                    return mlir::success(!!wg);
                  })
            .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
              return std::nullopt;
            });
    return result.value();
  };

  // Parse wg and sg attrs
  if (parser.parseCommaSeparatedList(parseElt))
    return {};

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  if (!wg && !sg) {
    parser.emitError(parser.getCurrentLocation(),
                     "Expecting at least one of sg and wg attributes.\n");
    return {};
  }

  return XeMapAttr::get(parser.getContext(), wg, sg);
}

void XeMapAttr::print(mlir::AsmPrinter &printer) const {
  bool printSep = false;
  printer << "<";
  if (getWg()) {
    printer << "wg = ";
    printWorkGroupMapAttrElements(printer, getWg().getSgLayout(),
                                  getWg().getSgData());
    printSep = true;
  }

  if (getSg()) {
    if (printSep)
      printer << ", ";
    printer << "sg = ";
    printSubGroupMapAttrElements(printer, getSg().getWiLayout(),
                                 getSg().getWiData(),
                                 getSg().getMmaBlockSize());
  }

  printer << ">";
}

} // namespace xegpu
} // namespace imex

#include <imex/Dialect/XeGPU/IR/XeGPUOpsDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/XeGPU/IR/XeGPUOpsAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/XeGPU/IR/XeGPUOpsTypes.cpp.inc>
