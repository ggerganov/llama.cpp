#!/usr/bin/env python3

"""
NOTE: The KLabels in this file must be kept up to date with the ones in the K semantics definition.
There is unfortunately no way to do this automatically.

This library provides a convenient interface to create KWasm programs in Kast format.
It is a mirror of the abstract syntax in the K semantics.
"""

from pyk.kast.inner import KSequence, KApply, KToken
from pyk.prelude.bytes import bytesToken
from pyk.utils import dequote_str

###########
# KLabels #
###########

MODULE = 'aModuleDecl'
MODULE_METADATA = 'moduleMeta'
TYPE = 'aTypeDefn'
FUNC_TYPE = 'aFuncType'
VEC_TYPE = 'aVecType'
VAL_TYPES = 'listValTypes'
VAL_TYPES_NIL = '.List{\"listValTypes\"}_ValTypes'
I32 = 'i32'
I64 = 'i32'
INTS = 'listInt'
INTS_NIL = '.List{\"listInt\"}_Ints'

FUNC = 'aFuncDefn'
FUNC_METADATA = 'funcMeta'

TABLE = 'aTableDefn'

MEMORY = 'aMemoryDefn'

GLOBAL = 'aGlobalDefn'
GLOBAL_TYPE = 'aGlobalType'

ELEM = 'aElemDefn'

DATA = 'aDataDefn'

START = 'aStartDefn'

IMPORT = 'aImportDefn'
FUNC_DESC = 'aFuncDesc'
GLOBAL_DESC = 'aGlobalDesc'
TABLE_DESC = 'aTableDesc'
MEMORY_DESC = 'aMemoryDesc'

EXPORT = 'aExportDefn'

DEFNS  = '___WASM-COMMON-SYNTAX_Defns_Defn_Defns'
INSTRS = '___WASM-COMMON-SYNTAX_Instrs_Instr_Instrs'

###################
# Basic Datatypes #
###################

def KInt(value : int):
    return KToken(str(value), 'Int')

def KFloat(value : float):
    return KToken(str(value), 'Float')

def KString(value : str):
    return KToken('"%s"' % value, 'String')

def KBytes(bs : bytes):
    # Change from python bytes repr to bytes repr in K.
    return bytesToken(dequote_str(str(bs))[2:-1])

###########
# Strings #
###########

def wasm_string(s : str):
    return KToken('\"%s\"' % s, 'WasmStringToken')

#########
# Lists #
#########

def KNamedList(klabel, empty_klabel, items):
    tail = KApply(empty_klabel, [])
    while not items == []:
        last = items.pop()
        tail = KApply(klabel, [last, tail])
    return tail

def defns(items):
    return KNamedList(DEFNS, EMPTY_STMTS, items)

def instrs(items):
    return KNamedList(INSTRS, EMPTY_STMTS, items)

def val_types(items):
    return KNamedList(VAL_TYPES, VAL_TYPES_NIL, items)

def ints(iis : [int]):
    kis = [KInt(x) for x in iis]
    return KNamedList(INTS, INTS_NIL, kis)

###########
# Empties #
###########

EMPTY_ID = KApply('.Identifier', [])
EMPTY_STMTS = '.List{\"listStmt\"}_EmptyStmts'
EMPTY_MAP = KApply('.Map', [])
EMPTY_OPT_STRING = KApply('.String', [])
EMPTY_DEFNS = KApply(EMPTY_STMTS, [])
EMPTY_FUNC_METADATA = KApply(FUNC_METADATA, [EMPTY_ID, EMPTY_MAP])

#########
# Types #
#########

i32 = KApply('i32', [])
i64 = KApply('i64', [])
f32 = KApply('f32', [])
f64 = KApply('f64', [])

MUT_CONST = KApply('mutConst', [])
MUT_VAR = KApply('mutVar', [])

def vec_type(valtypes):
    return KApply(VEC_TYPE, [valtypes])

def func_type(params, results):
    return KApply(FUNC_TYPE, [params, results])

def limits(tup):
    i = tup[0]
    j = tup[1]
    if j is None:
        return KApply('limitsMin', [KInt(i)])
    return KApply('limitsMinMax', [KInt(i), KInt(j)])

def global_type(mut, valtype):
    return KApply(GLOBAL_TYPE, [mut, valtype])

##########
# Instrs #
##########

  ########################
  # Control Instructions #
  ########################

NOP = KApply('aNop', [])
UNREACHABLE = KApply('aUnreachable', [])

def BLOCK(vec_type, instrs, block_info):
    return KApply('aBlock', [vec_type, instrs, block_info])

def IF(vec_type, then_instrs, else_instrs, block_info):
    return KApply('aIf', [vec_type, then_instrs, else_instrs, block_info])

def LOOP(vec_type, instrs, block_info):
    return KApply('aLoop', [vec_type, instrs, block_info])

RETURN = KApply('aReturn', [])

def BR(idx : int):
    return KApply('aBr', [KInt(idx)])

def BR_IF(idx : int):
    return KApply('aBr_if', [KInt(idx)])

def BR_TABLE(idxs : [int], default):
    return KApply('aBr_table', [ints(idxs + (default,))])

def CALL(function_idx : int):
    return KApply('aCall', [KInt(function_idx)])

def CALL_INDIRECT(type_idx : int):
    return KApply('aCall_indirect', [KInt(type_idx)])

  ##########################
  # Parametric Instruction #
  ##########################

DROP = KApply('aDrop', [])
SELECT = KApply('aSelect', [])

  ##############
  # Float UnOp #
  ##############

F32_ABS  = KApply('aFUnOp', [f32, KApply('aAbs', [])])
F32_CEIL = KApply('aFUnOp', [f32, KApply('aCeil', [])])
F32_FLOOR = KApply('aFUnOp', [f32, KApply('aFloor', [])])
F32_NEAREST = KApply('aFUnOp', [f32, KApply('aNearest', [])])
F32_NEG = KApply('aFUnOp', [f32, KApply('aNeg', [])])
F32_SQRT = KApply('aFUnOp', [f32, KApply('aSqrt', [])])
F32_TRUNC = KApply('aFUnOp', [f32, KApply('aTrunc', [])])
F64_ABS  = KApply('aFUnOp', [f64, KApply('aAbs', [])])
F64_CEIL = KApply('aFUnOp', [f64, KApply('aCeil', [])])
F64_FLOOR = KApply('aFUnOp', [f64, KApply('aFloor', [])])
F64_NEAREST = KApply('aFUnOp', [f64, KApply('aNearest', [])])
F64_NEG = KApply('aFUnOp', [f64, KApply('aNeg', [])])
F64_SQRT = KApply('aFUnOp', [f64, KApply('aSqrt', [])])
F64_TRUNC = KApply('aFUnOp', [f64, KApply('aTrunc', [])])

  ############
  # Int UnOp #
  ############

I32_CLZ = KApply('aIUnOp', [i32, KApply('aClz', [])])
I32_CTZ = KApply('aIUnOp', [i32, KApply('aCtz', [])])
I32_POPCNT = KApply('aIUnOp', [i32, KApply('aPopcnt', [])])
I64_CLZ = KApply('aIUnOp', [i64, KApply('aClz', [])])
I64_CTZ = KApply('aIUnOp', [i64, KApply('aCtz', [])])
I64_POPCNT = KApply('aIUnOp', [i64, KApply('aPopcnt', [])])

  ###############
  # Float BinOp #
  ###############

F32_ADD = KApply('aFBinOp', [f32, KApply('floatAdd', [])])
F32_SUB = KApply('aFBinOp', [f32, KApply('floatSub', [])])
F32_MUL = KApply('aFBinOp', [f32, KApply('floatMul', [])])
F32_DIV = KApply('aFBinOp', [f32, KApply('floatDiv', [])])
F32_MIN = KApply('aFBinOp', [f32, KApply('floatMin', [])])
F32_MAX = KApply('aFBinOp', [f32, KApply('floatMax', [])])
F32_COPYSIGN = KApply('aFBinOp', [f32, KApply('floatCopysign', [])])
F64_ADD = KApply('aFBinOp', [f64, KApply('floatAdd', [])])
F64_SUB = KApply('aFBinOp', [f64, KApply('floatSub', [])])
F64_MUL = KApply('aFBinOp', [f64, KApply('floatMul', [])])
F64_DIV = KApply('aFBinOp', [f64, KApply('floatDiv', [])])
F64_MIN = KApply('aFBinOp', [f64, KApply('floatMin', [])])
F64_MAX = KApply('aFBinOp', [f64, KApply('floatMax', [])])
F64_COPYSIGN = KApply('aFBinOp', [f64, KApply('floatCopysign', [])])

  #############
  # Int BinOp #
  #############

I32_ADD = KApply('aIBinOp', [i32, KApply('intAdd', [])])
I32_AND = KApply('aIBinOp', [i32, KApply('intAnd', [])])
I32_DIV_S = KApply('aIBinOp', [i32, KApply('intDiv_s', [])])
I32_DIV_U = KApply('aIBinOp', [i32, KApply('intDiv_u', [])])
I32_MUL = KApply('aIBinOp', [i32, KApply('intMul', [])])
I32_OR = KApply('aIBinOp', [i32, KApply('intOr', [])])
I32_REM_S = KApply('aIBinOp', [i32, KApply('intRem_s', [])])
I32_REM_U = KApply('aIBinOp', [i32, KApply('intRem_u', [])])
I32_ROTL = KApply('aIBinOp', [i32, KApply('intRotl', [])])
I32_ROTR = KApply('aIBinOp', [i32, KApply('intRotr', [])])
I32_SHL = KApply('aIBinOp', [i32, KApply('intShl', [])])
I32_SHR_S = KApply('aIBinOp', [i32, KApply('intShr_s', [])])
I32_SHR_U = KApply('aIBinOp', [i32, KApply('intShr_u', [])])
I32_SUB = KApply('aIBinOp', [i32, KApply('intSub', [])])
I32_XOR = KApply('aIBinOp', [i32, KApply('intXor', [])])
I64_ADD = KApply('aIBinOp', [i64, KApply('intAdd', [])])
I64_AND = KApply('aIBinOp', [i64, KApply('intAnd', [])])
I64_DIV_S = KApply('aIBinOp', [i64, KApply('intDiv_s', [])])
I64_DIV_U = KApply('aIBinOp', [i64, KApply('intDiv_u', [])])
I64_MUL = KApply('aIBinOp', [i64, KApply('intMul', [])])
I64_OR = KApply('aIBinOp', [i64, KApply('intOr', [])])
I64_REM_S = KApply('aIBinOp', [i64, KApply('intRem_s', [])])
I64_REM_U = KApply('aIBinOp', [i64, KApply('intRem_u', [])])
I64_ROTL = KApply('aIBinOp', [i64, KApply('intRotl', [])])
I64_ROTR = KApply('aIBinOp', [i64, KApply('intRotr', [])])
I64_SHL = KApply('aIBinOp', [i64, KApply('intShl', [])])
I64_SHR_S = KApply('aIBinOp', [i64, KApply('intShr_s', [])])
I64_SHR_U = KApply('aIBinOp', [i64, KApply('intShr_u', [])])
I64_SUB = KApply('aIBinOp', [i64, KApply('intSub', [])])
I64_XOR = KApply('aIBinOp', [i64, KApply('intXor', [])])

  ##########
  # TestOp #
  ##########

I32_EQZ = KApply('aTestOp', [i32, KApply('aEqz', [])])
I64_EQZ = KApply('aTestOp', [i64, KApply('aEqz', [])])

  #############
  # Int RelOp #
  #############

I32_EQ = KApply('aIRelOp', [i32, KApply('intEq', [])])
I32_NE = KApply('aIRelOp', [i32, KApply('intNe', [])])
I32_LT_U = KApply('aIRelOp', [i32, KApply('intLt_u', [])])
I32_GT_U = KApply('aIRelOp', [i32, KApply('intGt_u', [])])
I32_LT_S = KApply('aIRelOp', [i32, KApply('intLt_s', [])])
I32_GT_S = KApply('aIRelOp', [i32, KApply('intGt_s', [])])
I32_LE_U = KApply('aIRelOp', [i32, KApply('intLe_u', [])])
I32_GE_U = KApply('aIRelOp', [i32, KApply('intGe_u', [])])
I32_LE_S = KApply('aIRelOp', [i32, KApply('intLe_s', [])])
I32_GE_S = KApply('aIRelOp', [i32, KApply('intGe_s', [])])
I64_EQ = KApply('aIRelOp', [i64, KApply('intEq', [])])
I64_NE = KApply('aIRelOp', [i64, KApply('intNe', [])])
I64_LT_U = KApply('aIRelOp', [i64, KApply('intLt_u', [])])
I64_GT_U = KApply('aIRelOp', [i64, KApply('intGt_u', [])])
I64_LT_S = KApply('aIRelOp', [i64, KApply('intLt_s', [])])
I64_GT_S = KApply('aIRelOp', [i64, KApply('intGt_s', [])])
I64_LE_U = KApply('aIRelOp', [i64, KApply('intLe_u', [])])
I64_GE_U = KApply('aIRelOp', [i64, KApply('intGe_u', [])])
I64_LE_S = KApply('aIRelOp', [i64, KApply('intLe_s', [])])
I64_GE_S = KApply('aIRelOp', [i64, KApply('intGe_s', [])])

  ###############
  # Float RelOp #
  ###############

F32_LT = KApply('aFRelOp', [f32, KApply('floatLt', [])])
F32_GT = KApply('aFRelOp', [f32, KApply('floatGt', [])])
F32_LE = KApply('aFRelOp', [f32, KApply('floatLe', [])])
F32_GE = KApply('aFRelOp', [f32, KApply('floatGe', [])])
F32_EQ = KApply('aFRelOp', [f32, KApply('floatEq', [])])
F32_NE = KApply('aFRelOp', [f32, KApply('floatNe', [])])
F64_LT = KApply('aFRelOp', [f64, KApply('floatLt', [])])
F64_GT = KApply('aFRelOp', [f64, KApply('floatGt', [])])
F64_LE = KApply('aFRelOp', [f64, KApply('floatLe', [])])
F64_GE = KApply('aFRelOp', [f64, KApply('floatGe', [])])
F64_EQ = KApply('aFRelOp', [f64, KApply('floatEq', [])])
F64_NE = KApply('aFRelOp', [f64, KApply('floatNe', [])])

  ##############
  # Convert Op #
  ##############

I64_EXTEND_U_I32 = KApply('aCvtOp', [i64, KApply('aExtend_i32_u', [])])
I64_EXTEND_S_I32 = KApply('aCvtOp', [i64, KApply('aExtend_i32_s', [])])
I32_WRAP_I64 = KApply('aCvtOp', [i32, KApply('aWrap_i64', [])])

F64_PROMOTE_F32 = KApply('aCvtOp', [f64, KApply('aPromote_f32', [])])
F32_DEMOTE_F64 = KApply('aCvtOp', [f32, KApply('aDemote_f64', [])])

F32_CONVERT_U_I32 = KApply('aCvtOp', [f32, KApply('aConvert_i32_u', [])])
F64_CONVERT_U_I32 = KApply('aCvtOp', [f64, KApply('aConvert_i32_u', [])])
F32_CONVERT_U_I64 = KApply('aCvtOp', [f32, KApply('aConvert_i64_u', [])])
F64_CONVERT_U_I64 = KApply('aCvtOp', [f64, KApply('aConvert_i64_u', [])])
F32_CONVERT_S_I32 = KApply('aCvtOp', [f32, KApply('aConvert_i32_s', [])])
F64_CONVERT_S_I32 = KApply('aCvtOp', [f64, KApply('aConvert_i32_s', [])])
F32_CONVERT_S_I64 = KApply('aCvtOp', [f32, KApply('aConvert_i64_s', [])])
F64_CONVERT_S_I64 = KApply('aCvtOp', [f64, KApply('aConvert_i64_s', [])])

I32_TRUNC_U_F32 = KApply('aCvtOp', [i32, KApply('aTrunc_f32_u', [])])
I32_TRUNC_U_F64 = KApply('aCvtOp', [i32, KApply('aTrunc_f64_u', [])])
I64_TRUNC_U_F32 = KApply('aCvtOp', [i64, KApply('aTrunc_f32_u', [])])
I64_TRUNC_U_F64 = KApply('aCvtOp', [i64, KApply('aTrunc_f64_u', [])])
I32_TRUNC_S_F32 = KApply('aCvtOp', [i32, KApply('aTrunc_f32_s', [])])
I32_TRUNC_S_F64 = KApply('aCvtOp', [i32, KApply('aTrunc_f64_s', [])])
I64_TRUNC_S_F32 = KApply('aCvtOp', [i64, KApply('aTrunc_f32_s', [])])
I64_TRUNC_S_F64 = KApply('aCvtOp', [i64, KApply('aTrunc_f64_s', [])])

  #########
  # Const #
  #########

def F32_CONST(f: float):
    return KApply('aFConst', [f32, KFloat(f)])

def F64_CONST(f: float):
    return KApply('aFConst', [f64, KFloat(f)])

def I32_CONST(i : int):
    return (KApply('aIConst', [i32, KInt(i)]))

def I64_CONST(i : int):
    return (KApply('aIConst', [i64, KInt(i)]))

  #######################
  # Memory Instructions #
  #######################

def F32_STORE(offset : int):
    return KApply('aStore', [f32, KApply('storeOpStore', []), KInt(offset)])

def F64_STORE(offset : int):
    return KApply('aStore', [f64, KApply('storeOpStore', []), KInt(offset)])

def I32_STORE(offset : int):
    return KApply('aStore', [i32, KApply('storeOpStore', []), KInt(offset)])

def I64_STORE(offset : int):
    return KApply('aStore', [i64, KApply('storeOpStore', []), KInt(offset)])

def I32_STORE8(offset : int):
    return KApply('aStore', [i32, KApply('storeOpStore8', []), KInt(offset)])

def I64_STORE8(offset : int):
    return KApply('aStore', [i64, KApply('storeOpStore8', []), KInt(offset)])

def I32_STORE16(offset : int):
    return KApply('aStore', [i32, KApply('storeOpStore16', []), KInt(offset)])

def I64_STORE16(offset : int):
    return KApply('aStore', [i64, KApply('storeOpStore16', []), KInt(offset)])

def I64_STORE32(offset : int):
    return KApply('aStore', [i64, KApply('storeOpStore32', []), KInt(offset)])

def F32_LOAD(offset : int):
    return KApply('aLoad', [f32, KApply('loadOpLoad', []), KInt(offset)])

def F64_LOAD(offset : int):
    return KApply('aLoad', [f64, KApply('loadOpLoad', []), KInt(offset)])

def I32_LOAD(offset : int):
    return KApply('aLoad', [i32, KApply('loadOpLoad', []), KInt(offset)])

def I64_LOAD(offset : int):
    return KApply('aLoad', [i64, KApply('loadOpLoad', []), KInt(offset)])

def I32_LOAD16_S(offset : int):
    return KApply('aLoad', [i32, KApply('loadOpLoad16_s', []), KInt(offset)])

def I32_LOAD16_U(offset : int):
    return KApply('aLoad', [i32, KApply('loadOpLoad16_u', []), KInt(offset)])

def I64_LOAD16_S(offset : int):
    return KApply('aLoad', [i64, KApply('loadOpLoad16_s', []), KInt(offset)])

def I64_LOAD16_U(offset : int):
    return KApply('aLoad', [i64, KApply('loadOpLoad16_u', []), KInt(offset)])

def I32_LOAD8_S(offset : int):
    return KApply('aLoad', [i32, KApply('loadOpLoad8_s', []), KInt(offset)])

def I32_LOAD8_U(offset : int):
    return KApply('aLoad', [i32, KApply('loadOpLoad8_u', []), KInt(offset)])

def I64_LOAD8_S(offset : int):
    return KApply('aLoad', [i64, KApply('loadOpLoad8_s', []), KInt(offset)])

def I64_LOAD8_U(offset : int):
    return KApply('aLoad', [i64, KApply('loadOpLoad8_u', []), KInt(offset)])

def I64_LOAD32_U(offset : int):
    return KApply('aLoad', [i64, KApply('loadOpLoad32_u', []), KInt(offset)])

def I64_LOAD32_S(offset : int):
    return KApply('aLoad', [i64, KApply('loadOpLoad32_s', []), KInt(offset)])

MEMORY_GROW = KApply('aGrow', [])
MEMORY_SIZE = KApply('aSize', [])

  #######################
  # Global Instructions #
  #######################

def GET_GLOBAL(idx : int):
    return KApply('aGlobal.get', [KInt(idx)])

def SET_GLOBAL(idx : int):
    return KApply('aGlobal.set', [KInt(idx)])

  ######################
  # Local Instructions #
  ######################

def GET_LOCAL(idx : int):
    return KApply('aLocal.get', [KInt(idx)])

def SET_LOCAL(idx : int):
    return KApply('aLocal.set', [KInt(idx)])

def TEE_LOCAL(idx : int):
    return KApply('aLocal.tee', [KInt(idx)])

#######################
# Import Descriptions #
#######################

def func_desc(type : int, id=EMPTY_ID):
    return KApply(FUNC_DESC, [id, KInt(type)])

def global_desc(global_type, id=EMPTY_ID):
    return KApply(GLOBAL_DESC, [id, global_type])

def table_desc(lim, id=EMPTY_ID):
    return KApply(TABLE_DESC, [id, limits(lim)])

def memory_desc(lim, id=EMPTY_ID):
    return KApply(MEMORY_DESC, [id, limits(lim)])

################
# Declarations #
################

def type(func_type, metadata=EMPTY_ID):
    return KApply(TYPE, [func_type, metadata])

def func(type, locals, body, metadata=EMPTY_FUNC_METADATA):
    return KApply(FUNC, [type, locals, body, metadata])

def table(lim, metadata=EMPTY_ID):
    return KApply(TABLE, [limits(lim), metadata])

def memory(lim, metadata=EMPTY_ID):
    return KApply(MEMORY, [limits(lim), metadata])

def glob(type, init, metadata=EMPTY_ID):
    return KApply(GLOBAL, [type, init, metadata])

def elem(table_idx : int, offset, init : [int]):
    return KApply(ELEM, [KInt(table_idx), offset, ints(init)])

def data(memory_idx : int, offset, data : bytes):
    return KApply(DATA, [KInt(memory_idx), offset, KBytes(data)])

def start(start_idx : int):
    return KApply(START, [KInt(start_idx)])

def imp(mod_name, name, import_desc):
    return KApply(IMPORT, [mod_name, name, import_desc])

def export(name, index):
    return KApply(EXPORT, [name, KInt(index)])

def module_metadata(mid=None, fids=None, filename=None):
    # TODO: Implement module id and function ids metadata transformation.
    kfilename = EMPTY_OPT_STRING if filename is None else KString(filename)
    return KApply(MODULE_METADATA, [EMPTY_ID, EMPTY_MAP, kfilename])

EMPTY_MODULE_METADATA = module_metadata()

def module(types=EMPTY_DEFNS,
           funcs=EMPTY_DEFNS,
           tables=EMPTY_DEFNS,
           mems=EMPTY_DEFNS,
           globs=EMPTY_DEFNS,
           elem=EMPTY_DEFNS,
           data=EMPTY_DEFNS,
           start=EMPTY_DEFNS,
           imports=EMPTY_DEFNS,
           exports=EMPTY_DEFNS,
           metadata=EMPTY_MODULE_METADATA):
    """Construct a Kast of a module."""
    return KApply(MODULE,
                  [types,
                   funcs,
                   tables,
                   mems,
                   globs,
                   elem,
                   data,
                   start,
                   imports,
                   exports,
                   metadata])
