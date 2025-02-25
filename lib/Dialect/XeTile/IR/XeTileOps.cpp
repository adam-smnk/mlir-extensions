//===- XeTileOps.cpp - XeTile dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeTile dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace imex {
namespace xetile {

template <typename AttrType>
static mlir::ParseResult parseAttributeHelper(mlir::OpAsmParser &parser,
                                              mlir::OperationState &result,
                                              llvm::StringRef attrKeyword) {
  AttrType attr;
  mlir::Type ty;

  if (std::is_same<AttrType, mlir::Attribute>::value) {
    ty = mlir::Type{};
  } else if (std::is_same<AttrType, mlir::DenseI64ArrayAttr>::value) {
    ty = mlir::Type{};
  } else {
    assert(0 && "Unreachable.\n");
  }

  if (parser.parseCustomAttributeWithFallback(attr, ty))
    return mlir::failure();

  if (attr)
    result.addAttribute(attrKeyword, attr);
  return mlir::success();
}

static mlir::ParseResult
parseOptionalAttrDict(mlir::OpAsmParser &parser, mlir::OperationState &result,
                      llvm::ArrayRef<llvm::StringRef> allowedKeys) {

  // try to parse the left brace
  if (mlir::failed(parser.parseOptionalLBrace()))
    return mlir::success();

  auto parseElt = [&]() -> mlir::ParseResult {
    auto loc = parser.getCurrentLocation();
    llvm::StringRef nameId;
    if (parser.parseOptionalKeyword(&nameId, allowedKeys))
      return parser.emitError(loc, "invalid")
             << "attribute keyword: " << nameId << ".\n";

    if (parser.parseEqual())
      return mlir::failure();

    if (nameId == "transpose")
      return parseAttributeHelper<mlir::DenseI64ArrayAttr>(parser, result,
                                                           nameId);
    if (nameId == "padding") {
      return parseAttributeHelper<mlir::Attribute>(parser, result, nameId);
    }

    assert(0 && "Unreachable!");
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();

  if (parser.parseRBrace())
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult InitTileOp::verify() {

  // number of offsets must be 2 because init_tile creates 2D tiles
  // dynamic_offsets is always a subset of offsets, so checking this is
  // sufficient
  if (getStaticOffsets().size() != 2)
    return emitOpError("number of offsets must be 2");

  // if the source is a memref and has static shape, then dynamic shape and
  // strides arguments must not be present
  if (isSourceMemRef() && sourceMemRefHasStaticShape() &&
      (hasDynamicStrides() || hasDynamicShape()))
    return emitOpError("dynamic shape or strides are not allowed with a static "
                       "shaped memref as source");

  // if the source is a memref with dynamic shape, then a 2D dynamic shape
  // argument must be present
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getDynamicShape().size() != 2)
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic shape argument missing or it is not 2D");

  // if the source is a memref with dynamic shape, then a 2D dynamic strides
  // argument must be present
  if (isSourceMemRef() && !sourceMemRefHasStaticShape() &&
      getDynamicStrides().size() != 2)
    return emitOpError("memref with a dynamic shape is used as source but "
                       "dynamic strides argument missing or it is not 2D");

  // if the source is an address, the dynamic shape must be 2D
  if (isSourceInteger() && getDynamicShape().size() != 2)
    return emitOpError("address is used as source but dynamic shape argument "
                       "is missing or it is not 2D");

  // if the source is an address, dynamic strides must be 2D
  if (isSourceInteger() && getDynamicStrides().size() != 2)
    return emitOpError("address is used as source but dynamic strides argument "
                       "is missing or it is not 2D");

  return mlir::success();
}

void InitTileOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       xetile::TileType resultType, mlir::Value source,
                       llvm::ArrayRef<mlir::OpFoldResult> offsets) {
  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, resultType, source, dynamicOffsets, staticOffsets,
        mlir::ValueRange({}), /* empty dynamic shape*/
        mlir::ValueRange({})  /* empty dynamic strides*/
  );
}

void InitTileOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       xetile::TileType resultType, mlir::Value source,
                       llvm::ArrayRef<mlir::OpFoldResult> offsets,
                       llvm::ArrayRef<mlir::Value> dynamic_shape,
                       llvm::ArrayRef<mlir::Value> dynamic_strides) {
  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<mlir::Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, resultType, source, dynamicOffsets, staticOffsets,
        dynamic_shape, dynamic_strides);
}

mlir::ParseResult InitTileOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceRawOperands);
  llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  mlir::DenseI64ArrayAttr static_offsetsAttr;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4>
      dynamic_shapeOperands;
  llvm::SMLoc dynamic_shapeOperandsLoc;
  (void)dynamic_shapeOperandsLoc;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4>
      dynamic_stridesOperands;
  llvm::SMLoc dynamic_stridesOperandsLoc;
  (void)dynamic_stridesOperandsLoc;
  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceRawTypes);
  mlir::Type tileRawTypes[1];
  llvm::ArrayRef<mlir::Type> tileTypes(tileRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    auto odsResult =
        parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr);
    if (odsResult)
      return mlir::failure();
    result.getOrAddProperties<InitTileOp::Properties>().static_offsets =
        static_offsetsAttr;
  }
  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseLSquare())
      return mlir::failure();

    dynamic_shapeOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(dynamic_shapeOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return mlir::failure();
  }
  if (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseLSquare())
      return mlir::failure();

    dynamic_stridesOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(dynamic_stridesOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return mlir::failure();
  }

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(sourceRawTypes[0]))
    return mlir::failure();
  if (parser.parseArrow())
    return mlir::failure();

  if (parser.parseType(tileRawTypes[0]))
    return mlir::failure();
  llvm::copy(llvm::ArrayRef<int32_t>(
                 {1, static_cast<int32_t>(offsetsOperands.size()),
                  static_cast<int32_t>(dynamic_shapeOperands.size()),
                  static_cast<int32_t>(dynamic_stridesOperands.size())}),
             result.getOrAddProperties<InitTileOp::Properties>()
                 .operandSegmentSizes.begin());
  mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(tileTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0,
                             offsetsOperandsLoc, result.operands))
    return mlir::failure();
  if (parser.resolveOperands(dynamic_shapeOperands, odsBuildableType0,
                             dynamic_shapeOperandsLoc, result.operands))
    return mlir::failure();
  if (parser.resolveOperands(dynamic_stridesOperands, odsBuildableType0,
                             dynamic_stridesOperandsLoc, result.operands))
    return mlir::failure();
  return mlir::success();
}

void InitTileOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  printDynamicIndexList(printer, *this, getOffsets(), getStaticOffsetsAttr());
  if (!getDynamicShape().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getDynamicShape();
    printer << "]";
  }
  if (!getDynamicStrides().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getDynamicStrides();
    printer << "]";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getSource().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTile().getType();
}

mlir::LogicalResult LoadTileOp::verify() {
  auto transpose = getTransposeAttr();

  if (transpose && transpose.size() != 2) {
    return emitOpError("transpose must be two dimensional");
  }

  return mlir::success();
}

mlir::ParseResult LoadTileOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {

  mlir::OpAsmParser::UnresolvedOperand sourceTile;
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceTile);
  llvm::SMLoc sourceTileOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceTile))
    return mlir::failure();

  // try to parse the optional dictionary attributes
  if (parseOptionalAttrDict(parser, result, {"transpose", "padding"}))
    return mlir::failure();

  if (parser.parseColon())
    return mlir::failure();

  mlir::Type sourceType;
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceType);
  if (parser.parseType(sourceType))
    return mlir::failure();

  if (parser.parseArrow())
    return mlir::failure();

  mlir::Type valueType;
  llvm::ArrayRef<mlir::Type> outputValueTypes(valueType);
  if (parser.parseType(valueType))
    return mlir::failure();

  result.addTypes(outputValueTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceTileOperandLoc,
                             result.operands))
    return mlir::failure();
  return mlir::success();
}

static void printPaddingValue(mlir::Attribute paddingValue,
                              mlir::OpAsmPrinter &printer) {
  if (auto floatVal = llvm::dyn_cast<mlir::FloatAttr>(paddingValue)) {
    printer << floatVal.getValue() << " : " << floatVal.getType();
  } else if (auto intVal = llvm::dyn_cast<mlir::IntegerAttr>(paddingValue)) {
    printer << intVal.getValue() << " : " << intVal.getType();
  }
}

void LoadTileOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  bool printSep = false;

  printer << " { ";
  if ((*this)->getAttrs().size()) {
    if (getTransposeAttr()) {
      printer << "transpose = ";
      getTransposeAttr().print(printer);
      printSep = true;
    }
  }
  if (printSep)
    printer << ", ";
  printer << "padding = ";
  printPaddingValue(getPaddingValueOrDefault(), printer);
  printSep = true;

  printer << " } ";

  printer << " : ";
  printer << getSource().getType();
  printer << " -> ";
  printer << getValue().getType();
}

mlir::ParseResult TileMMAOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {

  mlir::OpAsmParser::UnresolvedOperand aRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> aOperands(aRawOperands);
  llvm::SMLoc aOperandsLoc;
  mlir::OpAsmParser::UnresolvedOperand bRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> bOperands(bRawOperands);
  llvm::SMLoc bOperandsLoc;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> cOperands;
  llvm::SMLoc cOperandsLoc;

  mlir::Type aRawTypes[1];
  llvm::ArrayRef<mlir::Type> aTypes(aRawTypes);
  mlir::Type bRawTypes[1];
  llvm::ArrayRef<mlir::Type> bTypes(bRawTypes);
  llvm::SmallVector<mlir::Type> cTypes;
  mlir::Type outputRawTypes[1];
  llvm::ArrayRef<mlir::Type> outputTypes(outputRawTypes);

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperands[0]))
    return mlir::failure();

  // try to parse optional C vector
  if (mlir::succeeded(parser.parseOptionalComma())) {
    cOperandsLoc = parser.getCurrentLocation();
    mlir::OpAsmParser::UnresolvedOperand operand;
    mlir::OptionalParseResult parseResult =
        parser.parseOptionalOperand(operand);

    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return mlir::failure();
      cOperands.push_back(operand);
    }
  }

  if (parser.parseColon())
    return mlir::failure();

  if (parser.parseType(aRawTypes[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  if (parser.parseType(bRawTypes[0]))
    return mlir::failure();

  if (mlir::succeeded(parser.parseOptionalComma())) {
    mlir::Type optionalType;
    mlir::OptionalParseResult parseResult =
        parser.parseOptionalType(optionalType);

    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return mlir::failure();
      cTypes.push_back(optionalType);
    }
  }

  if (parser.parseArrow())
    return mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return mlir::failure();

  result.addTypes(outputTypes);

  if (parser.resolveOperands(aOperands, aTypes, aOperandsLoc, result.operands))
    return mlir::failure();

  if (parser.resolveOperands(bOperands, bTypes, bOperandsLoc, result.operands))
    return mlir::failure();

  if (parser.resolveOperands(cOperands, cTypes, cOperandsLoc, result.operands))
    return mlir::failure();

  return mlir::success();
}

void TileMMAOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getA();
  printer << ", ";
  printer << getB();

  if (getC()) {
    printer << ", ";
    printer << getC();
  }
  printer << " : ";
  printer << getA().getType() << ", ";
  printer << getB().getType();
  if (getC()) {
    printer << ", ";
    printer << getC().getType();
  }
  printer << " -> ";
  printer << getOutput().getType();
}

mlir::LogicalResult TileMMAOp::verify() {
  int64_t aRank = getAType().getRank();
  int64_t bRank = getBType().getRank();

  mlir::Type aElemType = getAType().getElementType();
  mlir::Type bElemType = getBType().getElementType();
  mlir::Type outElemType = getOutput().getType().getElementType();

  auto aShape = getAType().getShape();
  auto bShape = getBType().getShape();
  auto outShape = getOutput().getType().getShape();

  // two vectors must have the same rank
  if (aRank != bRank)
    return emitOpError("A and B inputs must have the same rank.");

  // the two vector inputs must have the same element type
  if (aElemType != bElemType)
    return emitOpError("A and B inputs must have the same type.");

  if (getC() &&
      (llvm::cast<mlir::VectorType>(getC().getType()).getElementType() !=
       outElemType))
    return emitOpError("C and output vector must have the same type.");

  auto check4DMmaShapes = [](llvm::ArrayRef<int64_t> &A,
                             llvm::ArrayRef<int64_t> &B,
                             llvm::ArrayRef<int64_t> &Out) -> bool {
    return A[1] == B[0] && A[3] == B[2] && Out[0] == A[0] && Out[1] == B[1] &&
           Out[2] == A[2] && Out[3] == B[3];
  };

  auto check2DMmaShapes = [](llvm::ArrayRef<int64_t> &A,
                             llvm::ArrayRef<int64_t> &B,
                             llvm::ArrayRef<int64_t> &Out) -> bool {
    return A[1] == B[0] && Out[0] == A[0] && Out[1] == B[1];
  };

  // check mma shapes for 4D case
  if (aRank == 4 && !check4DMmaShapes(aShape, bShape, outShape))
    return emitOpError("incompatible A, B and output sizes for 4D tile mma op. "
                       "4D tile mma should have the shape (m x k x Bm x Bk) x "
                       "(k x n x Bk x Bn) = (m x n x Bm x Bn).");

  // check mma shape for 2D case
  if (aRank == 2 && !check2DMmaShapes(aShape, bShape, outShape))
    return emitOpError(
        "incompatible A, B and output sizes for 2D tile mma op. "
        "2D tile mma should have the shape (m x k) x (k x n) = (m x n).");

  // optional input C must have the same shape as output
  if (getC() &&
      llvm::cast<mlir::VectorType>(getC().getType()).getShape() != outShape)
    return emitOpError("input C must have the same shape as output.");

  return mlir::success();
}

} // namespace xetile
} // namespace imex

#define GET_OP_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOps.cpp.inc>
