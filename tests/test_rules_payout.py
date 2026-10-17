from src.rules import find_forbidden_language


def test_payable_language_is_flagged():
    assert find_forbidden_language("The amount payable is 9,500 EGP.")


def test_leaving_amount_after_deductible_is_flagged():
    assert find_forbidden_language("Deductible of 2,500 EGP, leaving $9,500 after deductible.")
    assert find_forbidden_language("leaving 9,500 EGP payable")


def test_will_be_paid_is_flagged():
    assert find_forbidden_language("The customer will be reimbursed next week.")


def test_neutral_deductible_statement_is_clean():
    assert find_forbidden_language("The policy deductible is 2,500 EGP.") == []
