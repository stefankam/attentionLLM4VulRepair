import helpers
import ctypes
from symbolicinput import SymbolicInput
from math import copysign

################### JUMP OPS #####################

def handleJumpOps(op, stack, items, symbols, symId):
  adr = stack.pop()
  out = (-1, False)
  if op == "JUMP":
    #jump to adr
    return jumpToLoc(adr, items)
    # if adr < 0:
    #   symbAdrJump(adr, 'Jump')
    # else:
    #   return jumpToLoc(adr, items)
  if op == "JUMPI":
    cond = stack.pop()
    if cond > 0:
      return jumpToLoc(adr, items)
    elif cond < 0:
      return symbAdrJump(cond, adr, symbols, items, symId)
  return out

def symbAdrJump(condition, address, symbols, items, symId):
  # if condition is symbolic, need to split paths
  sym = symbols[condition]
  del symbols[condition]
  # same for both paths
  x = SymbolicInput(symId[0], "IsZero", sym, address)
  symbols[symId[0]] = x
  symId[0] -= 1
  # tell path to split
  return (x.getId(), jumpToLoc(address, items)), -1

def makeJump(x, symbols, symId):
  # i.e. condition is not zero
  # new symbol mapping, because
  sym = symbols[x]
  del symbols[x]
  newSym = SymbolicInput(symId[0], "Not", sym)
  symbols[symId[0]] = newSym
  symId[0] -= 1

def jumpToLoc(adr, items):
  try:
    x = int(adr, 16)
  except TypeError:
    return -1, False
  return x, not isValidJumpTarget(x, items)

def isValidJumpTarget(loc, items):
  return loc not in invalidTargets and \
         helpers.convert(items[loc][0])[0] == "JUMPDEST"

invalidTargets = [2]

############### ARITHMETIC OPS #################

def handleArithOps(item, stack, symbols, symId):
  params = []
  for i in range(item[1]):
    p = stack.pop()
    if p >= 0:
      <fix/><fix/>params.append(int(p, 16))</fix></fix>
    else:
      <fix/><fix/>params.append(p)</fix>
  if params[0] >= 0 and params[1] >= 0:</fix>
    # if there are 3 params and the 3rd one is a symbol
    if len(params) == 3 and params[2] < 0:
      func = arithMapSym[item[0]]
      stack.append(func(params, symbols, symId))
    else:
      func = arithMap[item[0]]
      stack.append(helpers.toHex(func(params)))
  else:
    func = arithMapSym[item[0]]
    stack.append(func(params, symbols, symId))

def signedDiv(params, symbols, symId = -1):
  x = params[0]
  y = params[1]
  if not y:
    return y
  elif x == -(2**255) and y == -1:
    return -(2**255)
  else:
    return copysign(abs(x / y), x / y)

def signedMod(params, symbols, symId = -1):
  x = params[0]
  y = params[1]
  if y:
    return copysign(abs(x) % abs(y), x)
  return y

# need to check if this works
def signExtend(params, symbols, symId = -1):
  x = params[0]
  i = params[1]
  sign_bit = 1 << (i - 1)
  return (x & (sign_bit - 1)) - (x & sign_bit)

# Simple 2 argument operations involving symbols
def param2Simple(op, params, symbols, symId):
  if params[0] < 0:
    p0 = symbols[params[0]]
    del symbols[params[0]]
  else:
    p0 = params[0]

  if params[1] < 0:
    p1 = symbols[params[1]]
    del symbols[params[1]]
  else:
    p1 = params[1]

  x = SymbolicInput(symId[0], op, p0, p1)
  symbols[symId[0]] = x
  symId[0] -= 1
  return x.getId()

def param1Simple(op, params, symbols, symId):
  if params[0] < 0:
    p0 = symbols[params[0]]
    del symbols[params[0]]
  else:
    p0 = params[0]

  x = SymbolicInput(symId[0], op, p0, None)
  symbols[symId[0]] = x
  symId[0] -= 1
  return x.getId()

# Functions that take in 3 args for mods: add mod and mul mod
def mod3Arith(op, params, symbols, symId):

  # Get result of first operation
  if params[0] or params[1] < 0:
    sid = param2Simple(op, params[:2], symbols)
    p1p2 = symbols[sid]
    del symbols[sid]
  else:
    p1p2 = arithMap[op](params[:1])

  if params[2] < 0:
    p3 = symbols[params[2]]
    del symbols[params[2]]
  else:
    p3 = params[2]

  x = SymbolicInput(symId[0], 'Mod', p1p2, p3)
  symbols[symId[0]] = x
  symId[0] -= 1
  return x.getId()

# If no symbols
arithMap = {
  "ADD":        lambda params: params[0] + params[1],
  "MUL":        lambda params: params[0] * params[1],
  "SUB":        lambda params: params[0] - params[1],
  "DIV":        lambda params: params[0] / params[1] if params[1] else 0,
  "MOD":        lambda params: params[0] % params[1] if params[1] else 0,
  "ADDMOD":     lambda params: (params[0] + params[1]) % params[2] if params[2] else 0,
  "MULMOD":     lambda params: (params[0] * params[1]) % params[2] if params[2] else 0,
  "EXP":        lambda params: params[0] ** params[1],
  "SDIV":       signedDiv,
  "SMOD":       signedMod,
  "SIGNEXTEND": signExtend
}

# If there's a symbol
arithMapSym = {
  "ADD":        lambda params, symbols, symId: param2Simple('Add', params, symbols, symId),
  "MUL":        lambda params, symbols, symId: param2Simple('Mul', params, symbols, symId),
  "SUB":        lambda params, symbols, symId: param2Simple('Sub', params, symbols, symId),
  "DIV":        lambda params, symbols, symId: param2Simple('Div', params, symbols, symId),
  "MOD":        lambda params, symbols, symId: param2Simple('Mod', params, symbols, symId),
  "ADDMOD":     lambda params, symbols, symId: mod3Arith('Add', params, symbols, symId),
  "MULMOD":     lambda params, symbols, symId: mod3Arith('Mul', params, symbols, symId),
  "EXP":        lambda params, symbols, symId: param2Simple('Exp', params, symbols, symId),
  "SDIV":       signedDiv,
  "SMOD":       signedMod,
  "SIGNEXTEND": signExtend
}

############### BOOLEAN OPS #################

def makeUnsigned256(i):
    return ctypes.c_ubyte(i).value

def handleBoolOp(item, stack, symbols, symId):
  params = []
  for i in range(item[1]):
    p = stack.pop()
    if p >= 0:
      params.append(int(p, 16))
    else:
      params.append(p)

  # if only one arg
  if len(params) == 1:
    if params[0] < 0:
      func = boolMapSym[item[0]]
      stack.append(func(params, symbols, symId))
    else:
      func = boolMap[item[0]]
      stack.append(helpers.toHex(func(params)))
  elif params[0] >= 0 and params[1] >= 0: # 2 nonsymbolic args
    func = boolMap[item[0]]
    stack.append(helpers.toHex(func(params)))
  else:
    func = boolMapSym[item[0]]
    stack.append(func(params, symbols, symId))

def ltgt(op, params, symbols, symId):
  if params[0] < 0:
    p0 = symbols[params[0]]
    del symbols[params[0]]
  else:
    p0 = params[0]
    makeUnsigned256(p0)

  if params[1] < 0:
    p1 = symbols[params[1]]
    del symbols[params[1]]
  else:
    p1 = params[1]
    makeUnsigned256(p1)

  x = SymbolicInput(symId[0], op, p0, p1)
  symbols[symId[0]] = x
  symId[0] -= 1
  return x.getId()

# Boolmap for normal operations
boolMap = {
  "LT":     lambda params: makeUnsigned256(params[0]) < makeUnsigned256(params[1]),
  "GT":     lambda params: makeUnsigned256(params[0]) > makeUnsigned256(params[1]),
  "SLT":    lambda params: params[0] < params[1],
  "SGT":    lambda params: params[0] > params[1],
  "EQ":     lambda params: params[0] == params[1],
  "ISZERO": lambda params: not params[0],
  "AND":    lambda params: params[0] & params[1],
  "OR":     lambda params: params[0] | params[1],
  "XOR":    lambda params: params[0] ^ params[1],
  "NOT":    lambda params: params[0],
  "BYTE":   lambda params: (params[1] >> (8 * params[0])) & 0xFF
}

# Boolmap for operations with symbols
boolMapSym = {
  "LT":     lambda params, symbols, symId: ltgt('Lt', params, symbols, symId),
  "GT":     lambda params, symbols, symId: ltgt('Gt', params, symbols, symId),
  "SLT":    lambda params, symbols, symId: params[0] < params[1], #TODO
  "SGT":    lambda params, symbols, symId: params[0] > params[1], #TODO
  "EQ":     lambda params, symbols, symId: param2Simple('Eq', params, symbols, symId),
  "ISZERO": lambda params, symbols, symId: param1Simple('IsZero', params, symbols, symId),
  "AND":    lambda params, symbols, symId: param2Simple('And', params, symbols, symId),
  "OR":     lambda params, symbols, symId: param2Simple('Or', params, symbols, symId),
  "XOR":    lambda params, symbols, symId: param2Simple('Xor', params, symbols, symId),
  "NOT":    lambda params, symbols, symId: param1Simple('Not', params, symbols, symId),
  "BYTE":   lambda params, symbols, symId: (params[1] >> (8 * params[0])) & 0xFF #TODO
}

################ ENVIRONMENTAL OPS ##############

def handleEnvOps(item, stack, memory, symbols, userIn, symId):
  #func = envMap[item[0]]
  params = []
  for i in range(item[1]):
    p = stack.pop()
    if p >= 0:
      params.insert(0, int(p, 16))
    else:
      params.insert(0, p)
  if item[2] == 1:
    x = SymbolicInput(symId[0], 'id', None)
    symbols[symId[0]] = x
    stack.append(symId[0])
    userIn.append(symId[0])
    symId[0] -= 1
  #stack.append(helpers.toHex(func(params)))

############### DUP #############

def handleDupOp(op, symbols, stack, symId):
  num = int(op[3:])
  val = stack[-num]
  if val < 0:
    sym = symbols[val]
    x = SymbolicInput(symId[0], 'Dup', sym)
    symbols[symId[0]] = x
    symId[0] -= 1
    return x.getId()
  else:
    <fix/>return val</fix>

################ BLOCK OPS #################

def handleBlockOps(item, stack, symbols):
  pass
  # func = envMap[item[0]]
  # params = []
  # for i in range(item[1]):
    # params.insert(0, int(stack.pop(), 16))
  # stack.append(helpers.toHex(func(params)))

################ MEMORY OPS ##############

def handleMemoryOps(item, stack, memory, symbols):
  pass
  # func = envMap[item[0]]
  # params = []
  # for i in range(item[1]):
    # params.insert(0, int(stack.pop(), 16))
  # stack.append(helpers.toHex(func(params)))

################ STORAGE OPS ##############

def handleStorageOps(item, stack, storage, symbols, userIn):
  pass
  # func = envMap[item[0]]
  # params = []
  # for i in range(item[1]):
    # params.insert(0, int(stack.pop(), 16))
  # stack.append(helpers.toHex(func(params)))


#############################################