from TransferFunctions import TransferFunctions


def test_ReLu_Input_Greater_Than_0():
    ReLuOutput = TransferFunctions.ReLu(0.1)
    assert ReLuOutput == 0.1, "Should be 1 as when input is greater than 0 output = input"


def test_ReLu_Input_Less_Than_0():
    ReLuOutput = TransferFunctions.ReLu(-0.4)
    assert ReLuOutput == 0, "Should be 0 as when input is less than 0 all outputs should be 0"


