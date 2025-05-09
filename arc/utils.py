system_prompt = (
    "You are a wizard who solves Abstraction and Reasoning Corpus (ARC) puzzles. "
    "Infer the hidden rule from the examples and apply it to solve the test grid."
)

user_message_template1 = (
    "Below are 3 example input/output pairs. "
    "Study them to infer the underlying rule."
)

user_message_template2 = (
    "Now apply the rule to the following *test input* grid."
)

user_message_template3 = (
    "<think>First reason step by step about the pattern and how to apply it.</think>\n"
    "<answer>Please output *only* the final grid, with rows separated by new lines.</answer>"
)
