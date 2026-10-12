#pragma once

#include "test-case.h"

#include <string>
#include <vector>

//
// the runner: the single place that decides how a case is executed
//
//   - which cases run (the --test filter selects them, in table order)
//   - which models they run for (ctx_test::skip_reason)
//   - the KV cache mode a case inherits (the state cases keep the historical default of a
//     unified cache for a single sequence, the recurrent cases use the command line)
//   - whether it is repeated against a dirty cache (ctx_test::uses_fill)
//   - how a verdict is reported: a line per case in single model mode, a cell of the result
//     table in --models mode, and the reason of a case that did not pass always goes to the log
//     of the model
//

struct case_run {
    std::string name;
    test_status status = test_status::SKIP;
    std::string reason; // why the case was skipped or failed, empty when it passed
};

// the cases selected by the filter, all of them when it is empty, in table order
std::vector<const ctx_test *> select_cases(const std::string & filter);

// run the selected cases for one model and return one verdict per case - with report, print the
// verdict of every case as it finishes, for single model mode
std::vector<case_run> run_cases(model_run & mr, const std::vector<const ctx_test *> & cases, bool report);
