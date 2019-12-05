CHARACTER_DICT = set([
    u'A', u'C', u'E', u'D', u'G', u'F', u'I', u'H', u'K', u'M', u'L',
    u'N', u'Q', u'P', u'S', u'R', u'T', u'W', u'V', u'Y'
])
MAX_SEQUENCE_LENGTH = 46
MAX_MIC = 4
max_mic_buffer = 0.1

character_to_index = {
    (character): i
    for i, character in enumerate(CHARACTER_DICT)
}