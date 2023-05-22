import javascript
import semmle.javascript.PackageExports
import UMD
import semmle.javascript.security.dataflow.CodeInjectionCustomizations
import semmle.javascript.security.dataflow.CommandInjectionCustomizations
import semmle.javascript.security.dataflow.ReflectedXssCustomizations
import semmle.javascript.security.dataflow.TaintedPathCustomizations

/**
 * Holds if `name` is `default` or `exports`.
 */
private predicate isDefaultExportName(string name) { name in ["default", "exports"] }

/**
 * Holfds if `fn` is the constructor of a class named `className`.
 */
private predicate isConstructor(DataFlow::FunctionNode fn, string className) {
  exists(DataFlow::ClassNode cn |
    cn.getConstructor() = fn and
    className = cn.getName()
  )
}

/**
 * Gets a meaningful name for function `fn`, which must be exported by package
 * `pkg`.
 *
 * This is either the name of the function itself, or the name of the package if
 * the function is the only or default export of the package.
 */
private string getApiFunctionName(NpmPackage pkg, DataFlow::FunctionNode fn) {
  fn.getFile() = pkg.getAFile() and
  if not exists(fn.getName())
  then result = pkg.getPackageName()
  else
    if isConstructor(fn, _)
    then isConstructor(fn, result)
    else
      exists(string name | name = fn.getName() |
        if isDefaultExportName(name) then result = pkg.getPackageName() else result = name
      )
}

/**
 * Gets a parameter of a function exported by package `pkg`, where the name of
 * the function if `fnName` and the name of the parameter is `paramName`.
 *
 * We interpret "parameter" in a slightly loose sense here to include both the
 * function parameters per se as well as properties of these parameters.
 */
DataFlow::SourceNode getParameter(NpmPackage pkg, string fnName, string paramName) {
  exists(DataFlow::FunctionNode fn, DataFlow::ParameterNode param |
    param = getALibraryInputParameter() and
    param = fn.getAParameter() and
    fnName = getApiFunctionName(pkg, fn)
  |
    paramName = param.getName() and
    result = param
    or
    result = param.getAPropertyRead() and
    paramName = result.(DataFlow::PropRead).getPropertyName()
  )
}

/**
 * Holds if `nd` is a sink of the given `kind`.
 */
predicate isSinkOfKind(DataFlow::Node nd, string kind) {
  kind = "CodeInjection" and
  nd instanceof CodeInjection::Sink and
  // exclude low-fidelity sinks
  not nd =
    DataFlow::globalVarRef(["setImmediate", "setInterval", "setTimeout"])
        .getAnInvocation()
        .getArgument(0)
  or
  kind = "CommandInjection" and nd instanceof CommandInjection::Sink
  or
  kind = "ReflectedXss" and nd instanceof ReflectedXss::Sink
  or
  kind = "TaintedPath" and nd instanceof TaintedPath::Sink
}

/**
 * A configuration for tracking flow from API parameters to sinks.
 */
class CombinedConfig extends TaintTracking::Configuration {
  CombinedConfig() { this = "CombinedConfig" }

  override predicate isSource(DataFlow::Node nd) { nd = getParameter(_, _, _) }

  override predicate isSink(DataFlow::Node nd) { isSinkOfKind(nd, _) }

  override predicate isSanitizer(DataFlow::Node nd) {
    nd instanceof CodeInjection::Sanitizer
    or
    nd instanceof CommandInjection::Sanitizer
    or
    nd instanceof ReflectedXss::Sanitizer
    or
    nd instanceof TaintedPath::Sanitizer
    or
    // don't follow flow through property reads, since that changes the name of
    // the tracked entity
    nd instanceof DataFlow::PropRead
    or
    nd instanceof PropertyProjection
    or
    ArrayTaintTracking::arrayFunctionTaintStep(_, nd, _)
  }

  override predicate isSanitizerEdge(DataFlow::Node pred, DataFlow::Node succ) {
    pred = succ.asExpr().(LogicalAndExpr).getLeftOperand().flow()
  }
}
