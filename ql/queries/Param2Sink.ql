/**
 * @kind path-problem
 */

import Param2Sink
import DataFlow::PathGraph

from
  CombinedConfig cfg, DataFlow::PathNode source, DataFlow::PathNode sink,
  DataFlow::SourceNode param, NpmPackage pkg, string fnName, string paramName, string sinkKind
where
  cfg.hasFlowPath(source, sink) and
  source.getNode() = param and
  param = getParameter(pkg, fnName, paramName) and
  isSinkOfKind(sink.getNode(), sinkKind) and
  // Customise here
  paramName = "content" and
  sinkKind = "CodeInjection"
select source, source, sink,
  "Parameter " + paramName + " of function " + fnName + " in " + pkg.getPackageName() +
    " flows to " + sinkKind + " sink."
