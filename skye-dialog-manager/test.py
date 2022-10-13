# import spacy

# nlp =spacy.load('en_core_web_sm')
# temp_str = """
# Text: The original word text.
# Lemma: The base form of the word.(축약한 상태)
# POS: The simple part-of-speech tag.
# Tag: The detailed part-of-speech tag.
# Dep: Syntactic dependency, i.e. the relation between tokens.
# Shape: The word shape – capitalisation, punctuation, digits.
# is alpha: Is the token an alpha character?(알파벳만있느냐)
# is stop: Is the token part of a stop list, i.e. the most common words of the language?
# """.strip()
# print(temp_str)
# print("=="*40)
# str_format ="{:>10}"*8
# # print(str_format.format(*temp_dict.keys()))
# # print("=="*40)

# doc = nlp("what's your favorite sports my favorite sports is basball")
# for token in doc:
#     print(str_format.format(token.text, token.lemma_, token.pos_, token.tag_, 
#                             token.dep_, token.shape_, str(token.is_alpha), str(token.is_stop)))


k = "what is my favorite sports?"

if "food" in k or ("what" in k and "my" in k):
    print("success")
else:
    print("error")