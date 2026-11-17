import math

from nlolib._config import _lower_operators
from nlolib._runtime_expr import (
    RUNTIME_CONTEXT_DISPERSION,
    RUNTIME_CONTEXT_DISPERSION_FACTOR,
    RUNTIME_CONTEXT_LINEAR,
    RUNTIME_CONTEXT_LINEAR_FACTOR,
    RUNTIME_CONTEXT_POTENTIAL,
    RUNTIME_CONTEXT_NONLINEAR,
    translate_callable,
)


def _assert_raises_value_error(func, expected_substring):
    try:
        func()
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert expected_substring in str(exc), f"missing '{expected_substring}' in '{exc}'"


def test_numpy_functions_translate():
    alpha = 0.25
    # Translation is syntax-based, so NumPy-style qualified calls are validated
    # without requiring numpy to be installed in the test environment.
    translated = translate_callable(
        lambda A, w: 1j * (np.sin(w) + alpha * np.cos(w)),  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert "sin(" in translated.expression
    assert "cos(" in translated.expression
    assert translated.constants == [alpha]


def test_math_functions_translate():
    offset = 1.0
    translated = translate_callable(
        lambda A, w: 1j * (math.sqrt((w * w) + offset) + math.log((w * w) + offset)),  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert "sqrt(" in translated.expression
    assert "log(" in translated.expression
    assert translated.constants == [offset]


def test_beta_sum_translation():
    beta2 = 0.5
    beta3 = -0.02
    beta4 = 0.003
    translated = translate_callable(
        lambda A, w: 1j * (beta2 * (w**2) + beta3 * (w**3) + beta4 * (w**4)),  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert "^2" in translated.expression
    assert "^3" in translated.expression
    assert "^4" in translated.expression
    assert translated.constants == [beta2, beta3, beta4]


def test_diffraction_factor_translation():
    beta_t = -0.015
    translated = translate_callable(
        lambda A, w: 1j * beta_t * w,  # noqa: E731
        RUNTIME_CONTEXT_DISPERSION_FACTOR,
    )
    assert translated.expression.count("w") > 0
    assert translated.constants == [beta_t]


def test_raman_like_nonlinear_translation():
    gamma = 0.01
    f_r = 0.18
    translated = translate_callable(
        lambda A, I, V: 1j * A * (gamma * (1.0 - f_r) * I + gamma * f_r * V),  # noqa: E731
        RUNTIME_CONTEXT_NONLINEAR,
    )
    assert "A" in translated.expression
    assert "I" in translated.expression
    assert "V" in translated.expression
    assert translated.constants == [gamma, f_r]


def test_tensor_linear_factor_symbols_translate():
    beta2 = 0.02
    beta_t = -0.01
    translated = translate_callable(
        lambda A, wt, kx, ky: 1j * (beta2 * (wt**2) + beta_t * ((kx**2) + (ky**2))),  # noqa: E731
        RUNTIME_CONTEXT_LINEAR_FACTOR,
    )
    assert "wt" in translated.expression
    assert "kx" in translated.expression
    assert "ky" in translated.expression
    assert translated.constants == [beta2, beta_t]


def test_potential_symbols_translate():
    gx = 2.0e-4
    gy = 1.5e-4
    translated = translate_callable(
        lambda x, y: gx * (x**2) + gy * (y**2),  # noqa: E731
        RUNTIME_CONTEXT_POTENTIAL,
    )
    assert "x" in translated.expression
    assert "y" in translated.expression
    assert translated.constants == [gx, gy]


def test_rejects_unsupported_numpy_call():
    _assert_raises_value_error(
        lambda: translate_callable(  # noqa: E731
            lambda A, w: np.tanh(w),  # noqa: E731
            RUNTIME_CONTEXT_DISPERSION_FACTOR,
        ),
        "unsupported function call",
    )


def test_rejects_invalid_signature():
    _assert_raises_value_error(
        lambda: translate_callable(  # noqa: E731
            lambda: 0.0,  # noqa: E731
            RUNTIME_CONTEXT_DISPERSION_FACTOR,
        ),
        "dispersion_factor operator must take",
    )


def test_rejects_disallowed_complex_literal():
    _assert_raises_value_error(
        lambda: translate_callable(  # noqa: E731
            lambda A, w: (2.0 + 3.0j) * w,  # noqa: E731
            RUNTIME_CONTEXT_DISPERSION_FACTOR,
        ),
        "complex literals",
    )


def test_w_and_wt_name_the_same_axis():
    # The linear operator of a temporal run lowers in dispersion_factor (w), of
    # a tensor run in linear_factor (wt); either spelling works in both.
    assert translate_callable(lambda A, wt: wt, RUNTIME_CONTEXT_DISPERSION_FACTOR).expression == "w"
    assert translate_callable(lambda A, w: w, RUNTIME_CONTEXT_LINEAR_FACTOR).expression == "wt"
    assert translate_callable(lambda A, D, h, w: w, RUNTIME_CONTEXT_LINEAR).expression == "wt"
    assert translate_callable(lambda A, D, h, wt: wt, RUNTIME_CONTEXT_DISPERSION).expression == "w"


def test_parameters_bind_by_name_then_by_order():
    # field and omega are not symbols, so they take the next unused symbols.
    assert translate_callable(lambda field, omega: field * omega,
                              RUNTIME_CONTEXT_DISPERSION_FACTOR).expression == "(A*w)"
    # A named parameter keeps its symbol wherever it sits.
    assert translate_callable(lambda q, A: q * A, RUNTIME_CONTEXT_DISPERSION_FACTOR).expression == "(w*A)"
    assert translate_callable(lambda u, v: u * v, RUNTIME_CONTEXT_POTENTIAL).expression == "(x*y)"
    # (b) named parameters bind first, the rest fill the remaining symbols in order.
    assert translate_callable(lambda B, A: B * A, RUNTIME_CONTEXT_DISPERSION_FACTOR).expression == "(w*A)"
    assert translate_callable(lambda q, D, A: q * D, RUNTIME_CONTEXT_DISPERSION).expression == "(h*D)"


def test_parameter_naming_a_missing_symbol_is_rejected():
    # (c) kx is a runtime symbol, but not one dispersion_factor provides.
    _assert_raises_value_error(
        lambda: translate_callable(lambda A, kx: kx, RUNTIME_CONTEXT_DISPERSION_FACTOR),  # noqa: E731
        "parameter 'kx' names a symbol the dispersion_factor operator does not provide",
    )
    _assert_raises_value_error(
        lambda: translate_callable(lambda x, wt: x, RUNTIME_CONTEXT_POTENTIAL),  # noqa: E731
        "parameter 'wt' names a symbol the potential operator does not provide",
    )


def test_two_parameters_naming_one_symbol_are_rejected():
    # (d) w and wt are the same axis, so naming both is ambiguous.
    _assert_raises_value_error(
        lambda: translate_callable(lambda A, w, wt: w, RUNTIME_CONTEXT_LINEAR_FACTOR),  # noqa: E731
        "both name 'wt'",
    )
    _assert_raises_value_error(
        lambda: translate_callable(lambda A, wt, w: w, RUNTIME_CONTEXT_DISPERSION),  # noqa: E731
        "both name 'w'",
    )


def test_context_symbols_match_the_c_parser():
    # The C parser rejects kx, ky, x, y and t in the temporal contexts and wt,
    # kx, ky in potential; the binding reports them before the C call.
    # (f) Symbols enter only through parameters: a free name in the body is a
    # captured constant whatever it is called, so an unbound one is an error
    # and a bound one is a constant, never the runtime symbol.
    _assert_raises_value_error(
        lambda: translate_callable(lambda A, I: kz * A, RUNTIME_CONTEXT_NONLINEAR),  # noqa: E731,F821
        "unknown identifier 'kz'",
    )
    _assert_raises_value_error(
        lambda: translate_callable(lambda A, I: wt * A, RUNTIME_CONTEXT_NONLINEAR),  # noqa: E731,F821
        "unknown identifier 'wt'",
    )
    w = 3.0
    translated = translate_callable(lambda A, I: w * A, RUNTIME_CONTEXT_NONLINEAR)  # noqa: E731
    assert translated.expression == "(c0*A)" and translated.constants == [3.0]
    kx = 0.5
    translated = translate_callable(lambda A, w: kx * w, RUNTIME_CONTEXT_DISPERSION_FACTOR)  # noqa: E731
    assert translated.expression == "(c0*w)" and translated.constants == [0.5]
    _assert_raises_value_error(
        lambda: translate_callable(lambda A, I, V, x: x, RUNTIME_CONTEXT_NONLINEAR),  # noqa: E731
        "between 1 and 3",
    )


def test_lambdas_sharing_a_line_lower_to_their_own_bodies():
    # getsource returns the whole line; each lambda must lower to its own body.
    ops = (lambda A, w: 1j * 2.0 * w * w, lambda A, I: 1j * 3.0 * A * I)  # noqa: E731
    assert translate_callable(ops[0], RUNTIME_CONTEXT_DISPERSION_FACTOR).expression == "(((i*2)*w)*w)"
    assert translate_callable(ops[1], RUNTIME_CONTEXT_NONLINEAR).expression == "(((i*3)*A)*I)"

    def make(a, b):
        return (lambda A, w: a * w, lambda A, w: b * w)  # noqa: E731

    first, second = make(1.5, 2.5)
    assert translate_callable(first, RUNTIME_CONTEXT_DISPERSION_FACTOR).constants == [1.5]
    assert translate_callable(second, RUNTIME_CONTEXT_DISPERSION_FACTOR).constants == [2.5]

    table = {
        "linear": lambda A, w: 4.0 * w,  # noqa: E731
        "nonlinear": lambda A, I: 5.0 * I,  # noqa: E731
    }
    assert translate_callable(table["nonlinear"], RUNTIME_CONTEXT_NONLINEAR).expression == "(5*I)"


def test_constant_bindings_win_over_captured_names():
    gamma = 1.0
    translated = translate_callable(
        lambda A, I: 1j * gamma * A * I,  # noqa: E731
        RUNTIME_CONTEXT_NONLINEAR,
        constant_bindings={"gamma": 2.5},
    )
    assert translated.constants == [2.5]


def test_operators_lower_to_the_field_the_run_type_compiles():
    operators = {
        "linear_operator": lambda A, w: 1j * w,
        "linear_step_operator": lambda A, D, h: D * h,
        "potential_operator": None,
        "nonlinear_operator": lambda A, I: 1j * A * I,
    }
    temporal, _ = _lower_operators(operators, tensor_run=False, constant_bindings=None)
    assert set(temporal) == {"dispersion_factor_expr", "dispersion_expr", "nonlinear_expr"}
    assert temporal["dispersion_factor_expr"] == "(i*w)"

    tensor_ops = dict(operators, potential_operator=lambda x, y: x * y)
    tensor, _ = _lower_operators(tensor_ops, tensor_run=True, constant_bindings=None)
    assert set(tensor) == {"linear_factor_expr", "linear_expr", "potential_expr", "nonlinear_expr"}
    assert tensor["linear_factor_expr"] == "(i*wt)"

    _assert_raises_value_error(
        lambda: _lower_operators(tensor_ops, tensor_run=False, constant_bindings=None),  # noqa: E731
        "only used by tensor runs",
    )


def test_constants_from_several_operators_are_renumbered():
    a, b = 0.5, 0.25
    expressions, constants = _lower_operators(
        {"linear_operator": lambda A, w: a * w, "nonlinear_operator": lambda A, I: b * I},
        tensor_run=False,
        constant_bindings=None,
    )
    assert expressions == {"dispersion_factor_expr": "(c0*w)", "nonlinear_expr": "(c1*I)"}
    assert constants == [a, b]


def main():
    test_numpy_functions_translate()
    test_math_functions_translate()
    test_beta_sum_translation()
    test_diffraction_factor_translation()
    test_raman_like_nonlinear_translation()
    test_tensor_linear_factor_symbols_translate()
    test_potential_symbols_translate()
    test_rejects_unsupported_numpy_call()
    test_rejects_invalid_signature()
    test_rejects_disallowed_complex_literal()
    test_w_and_wt_name_the_same_axis()
    test_parameters_bind_by_name_then_by_order()
    test_parameter_naming_a_missing_symbol_is_rejected()
    test_two_parameters_naming_one_symbol_are_rejected()
    test_context_symbols_match_the_c_parser()
    test_lambdas_sharing_a_line_lower_to_their_own_bodies()
    test_constant_bindings_win_over_captured_names()
    test_operators_lower_to_the_field_the_run_type_compiles()
    test_constants_from_several_operators_are_renumbered()
    print("test_python_runtime_expr_translation: callable translation coverage validated.")


if __name__ == "__main__":
    main()
