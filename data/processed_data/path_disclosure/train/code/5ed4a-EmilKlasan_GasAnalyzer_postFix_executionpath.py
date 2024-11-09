import helpers
import ophandlers
import oplists
from copy import deepcopy

class ExecutionPath:
  def __init__(self, opcodes,
                     functions,
                     stack    = [],
                     memory   = [],
                     storage  = {},
                     symbols  = {},
                     userIn   = [],
                     instrPtr = 1,
                     symId    = [-1]):
    self.opcodes   = opcodes
    self.functions = functions
    self.stack     = stack
    self.memory    = memory
    self.storage   = storage
    self.symbols   = symbols
    self.userIn    = userIn  # list of symbols that directly represent user input
    self.instrPtr  = instrPtr # instruction pointer
    self.symId     = symId

  def takeJumpPath(self, pair, symbols):
    # negate isZero, set new symbol, check/return for jumpdest
    self.instrPtr = pair[1][0]
    <fix/>ophandlers.makeJump(pair[0], symbols, self.symId)</fix>

  def traverse(self, pathSymbols):
    gasCost = 0
    stop    = False
    while not stop:
      if self.instrPtr == 2:
        break
      item = self.opcodes[self.instrPtr]
      op   = item[0]
      if op in oplists.terminalOps:
        break
      elif op in oplists.arithOps:
        ophandlers.handleArithOps(item,
                                  self.stack,
                                  self.symbols,
                                  self.symId)
      elif op in oplists.boolOps:
        ophandlers.handleBoolOp(item,
                                self.stack,
                                self.symbols,
                                self.symId)
      elif op == "SHA3":
        pass
      elif op in oplists.envOps:
        ophandlers.handleEnvOps(item,
                                self.stack,
                                self.memory,
                                self.symbols,
                                self.userIn,
                                self.symId)
      elif op in oplists.blockOps:
        ophandlers.handleBlockOps(item,
                                  self.stack,
                                  self.symbols)
      elif op in oplists.jumpOps:
        result  = ophandlers.handleJumpOps(op,
                                           self.stack,
                                           self.opcodes,
                                           self.symbols,
                                           self.symId)
        if result[0] != -1 and result[1] != -1:
          self.instrPtr, stop = result
          continue
        elif result[1] == -1:
          # symbolic, need to split
          # pass path-specific parameters by value
          ep1 = ExecutionPath(self.opcodes,
                              self.functions,
                              self.stack[:],
                              self.memory[:],
                              deepcopy(self.storage),
                              deepcopy(self.symbols),
                              self.userIn[:],
                              self.instrPtr)
          <fix/>ep1.takeJumpPath(result[0], self.symbols)
          self.instrPtr = item[-1]
          print("splitting")
          return [self, ep1]</fix>
      elif op in oplists.memOps:
        ophandlers.handleMemoryOps(item,
                                   self.stack,
                                   self.memory,
                                   self.symbols)
      elif op in oplists.storOps:
        ophandlers.handleStorageOps(item,
                                    self.stack,
                                    self.storage,
                                    self.symbols,
                                    self.userIn)
      elif op == "JUMPDEST":
        pass
      elif op == "POP":
        self.stack.pop()
      elif op == "PC":
        self.stack.append(i)
      elif op[:4] == "PUSH":
        # push value to stack
        self.stack.append(op[7:])
      elif op[:3] == "DUP":
        <fix/>on = ophandlers.handleDupOp(op,
                                    self.symbols,
                                    self.stack,
                                    self.symId)
        self.stack.append(on)</fix>
      elif op[:4] == "SWAP":
        num         = int(op[4:])
        tmp         = self.stack[-num]
        self.stack[-num] = self.stack[-1]
        self.stack[-1]   = tmp
      elif op[:3] == "LOG":
        pass

      # print(self)
      # print item
      # print("Stack:")
      # helpers.prettyPrint(self.stack)
      # print("Symbols")
      # for x in self.symbols:
      #   print ''
      #   print 'symbol {}:'.format(x)
      #   print self.symbols[x].derive()
      # print ''

      #gasCost = gasCost + calculateGasCost(item[0])
      self.instrPtr = item[-1]
      <fix/>stop = self.instrPtr == -1</fix>
    # print("Stack:")
    # helpers.prettyPrint(self.stack)
    # print("Memory:")
    # helpers.prettyPrint(self.memory)
    # print("Storage:")
    # helpers.prettyPrint(self.storage)

    # for x in self.symbols:
    #   print ''
    #   print 'symbol {}:'.format(x)
    #   print self.symbols[x].derive()
    pathSymbols.append(self.symbols)
    return []

