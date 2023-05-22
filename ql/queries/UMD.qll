import javascript
private import semmle.javascript.internal.CachedStages

class UmdModule extends Module {
  DataFlow::FunctionNode factory;

  UmdModule() {
    Stages::DataFlowStage::ref() and
    exists(ImmediatelyInvokedFunctionExpr iife, DataFlow::PropWrite modExport |
      this = iife.getContainer() and
      iife.getArgument(_).flow().getALocalSource() = factory and
      modExport.getContainer() = iife and
      modExport.getBase().asExpr().(GlobalVarAccess).getName() = "module" and
      modExport.getPropertyName() = "exports" and
      modExport.getRhs() = factory.getACall()
    )
  }

  override DataFlow::Node getAnExportedValue(string name) {
    exists(DataFlow::SourceNode res |
      factory.getReturnNode().getALocalSource() = res and
      result = res.getAPropertyWrite(name).getRhs()
    )
  }
}
