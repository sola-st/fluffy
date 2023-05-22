/**
 * @name Flow from parameter to known sinks
 * @kind table
 */

import Param2Sink

predicate parameterFlowsToSink(DataFlow::Node p, string kind) {
  exists(
    CombinedConfig cfg, DataFlow::SourcePathNode source, DataFlow::MidPathNode last,
    DataFlow::SinkPathNode sink
  |
    source.wraps(p, cfg) and
    last = source.getASuccessor*() and
    last.getPathSummary().hasReturn() = false and
    sink = last.getASuccessor() and
    isSinkOfKind(sink.getNode(), kind)
  )
  or
  exists(CombinedConfig cfg, DataFlow::Node snk |
    isSinkOfKind(snk, kind) and
    p = snk.getALocalSource() and
    cfg.isSource(p)
  )
}

/**
 * A class definition, viewed as a documentable entity.
 */
private class DocumentableClassDefinition extends Documentable, ClassDefinition { }

/**
 * Gets the doc comment attached to the function to which `p` belongs.
 *
 * For constructor parameters, we look for the doc comment attached to the
 * enclosing class (but only if the constructor is not documented).
 */
JSDoc getDocComment(Parameter p) {
  if exists(p.getDocumentation())
  then result = p.getDocumentation()
  else
    exists(Function f | p = f.getAParameter() |
      if exists(f.getDocumentation())
      then result = f.getDocumentation()
      else
        exists(DocumentableClassDefinition cd | f = cd.getConstructor().getBody() |
          result = cd.getDocumentation()
        )
    )
}

/**
 * Gets the documentation for parameter `p`, if any.
 */
string getDocumentation(DataFlow::ParameterNode p) {
  exists(JSDoc doc | doc = getDocComment(p.getParameter()) |
    exists(JSDocParamTag tag, string name |
      tag.getParent() = doc and name = tag.getName() and tag.getName() = p.getName()
    |
      result = tag.getDescription()
    )
  )
}

string getDocumentationOpt(DataFlow::SourceNode p) {
  if exists(getDocumentation(p)) then result = getDocumentation(p) else result = ""
}

from NpmPackage pkg, string fn, string pn, DataFlow::Node p, string kind, File file, int line
where
  p = getParameter(pkg, fn, pn) and
  (
    parameterFlowsToSink(p, kind)
    or
    not parameterFlowsToSink(p, _) and
    kind = "None"
  ) and
  p.hasLocationInfo(file.getAbsolutePath(), line, _, _, _)
select pkg.getPackageName(), pkg.getPackageJson().getVersion(), fn, pn, kind,
  getDocumentationOpt(p), file.getRelativePath(), line
