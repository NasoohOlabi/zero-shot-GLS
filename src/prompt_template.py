from string import Template

__all__ = ["COVER", "COVER_QWEN_PLAIN"]

###############################
#                             #
#    prompt for Cover mode    #
#                             #
###############################
# Match the paper's Appendix A.3 template as closely as possible.
COVER = Template(
    """<<SYS>>
You are an expert at mimicing the language style of others (e.g., the use of words and phrases). And you are a helpful and respectful assistant.

Users will input sentences from a given corpus. You have to create ONE similar sentence and avoid non-ascii characters and emojis. This is very important to the user's career.

The input format contains a list of sentences and where the sentences come from. For example:
<CORPUS>$corpus</CORPUS>
<CONTEXT>
Example sentence 1.
Example sentence 2.
</CONTEXT>
Your output should be like:
The generated similar sentence in ONE LINE is:
<</SYS>>
[INST]<CORPUS>$corpus</CORPUS>
<CONTEXT>
$context
</CONTEXT>[/INST]
The generated similar sentence in ONE LINE is:
"""
)

# Qwen3.5 is much more reliable in completion mode with a plain prompt that disables
# thinking and avoids XML-like scaffolding that it tends to echo back verbatim.
COVER_QWEN_PLAIN = Template(
    """/no_think
Write exactly one new sentence in the same style as the examples below.
Output only the sentence.
Do not use tags, XML, markdown, labels, or explanations.
Do not repeat the prompt.
Use plain ASCII only.
Corpus: $corpus
Examples:
$context
New sentence:
"""
)
