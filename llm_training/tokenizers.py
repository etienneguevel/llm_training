from collections import Counter

def byte_pairs(text: str):
    text_bytes = list(text.encode("utf-8"))
    pairs = list(zip(text_bytes[:-1], text_bytes[1:]))

    return Counter(pairs)

def merge_ids(text, ids, idx):

    new_text = []
    pair_wait = False
    for i, el in enumerate(text[:-1]):
        if pair_wait:
            pair_wait = False
            continue

        if (el, text[i+1]) == ids:
            new_text.append(idx)
            pair_wait = True

        else:
            new_text.append(el)

    if not pair_wait:
        new_text.append(text[-1])

    return new_text


if __name__ == "__main__":
    example = "My favourite word is strawberry or abracadabra."
    text_bytes = list(example.encode("utf-8"))
    print(text_bytes)
    pairs = byte_pairs(example)
    mostc = pairs.most_common(1)[0][0]
    print(mostc)

    new_text = merge_ids(text_bytes, mostc, 256)
    print(new_text)
