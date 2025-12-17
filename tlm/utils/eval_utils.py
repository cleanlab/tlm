from tlm.types import SemanticEval


def group_evals(evals: list[SemanticEval] | None) -> tuple[list[SemanticEval], list[SemanticEval]]:
    if evals is None:
        return [], []

    evals_requiring_response = []
    evals_not_requiring_response = []

    for eval in evals:
        if eval.response_identifier is not None:
            evals_requiring_response.append(eval)
        else:
            evals_not_requiring_response.append(eval)

    return evals_requiring_response, evals_not_requiring_response
