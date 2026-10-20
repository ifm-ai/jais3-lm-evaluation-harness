"""Utilities for the Arabic linguistic I'rab and Tashkeel generation benchmarks."""

import re

CHOICE_LETTERS = ["A", "B", "C", "D"]
BRACKETED_LABEL_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")


def normalize_label(label):
    return str(label).strip().lower()


def extract_bracketed_label(resps, docs):
    def extract(resp):
        if not isinstance(resp, str):
            return "[invalid]"

        matches = BRACKETED_LABEL_RE.findall(resp)
        if not matches:
            return "[invalid]"

        return matches[-1]

    return map(lambda r: extract(r[0] if r else ""), resps)


def exact_match_normalized_label(predictions, references, **kwargs):
    prediction = predictions[0] if predictions else "[invalid]"
    reference = references[0] if references else "[invalid]"
    return {
        "exact_match": float(
            normalize_label(prediction) == normalize_label(reference)
        )
    }


def build_prompt(instruction_lang: str, labels_lang: str, num_choices: int) -> str:
    if instruction_lang not in ["en", "ar"]:
        raise ValueError("instruction_lang must be 'en' or 'ar'")
    if labels_lang not in ["en", "ar"]:
        raise ValueError("labels_lang must be 'en' or 'ar'")
    if num_choices < 2 or num_choices > 26:
        raise ValueError("num_choices must be between 2 and 26")

    letters_en = [chr(ord("A") + i) for i in range(num_choices)]
    letters_ar = [
        "أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي",
        "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر",
        "ش", "ت", "ث", "خ", "ذ", "ض",
    ][:num_choices]
    letters = letters_en if labels_lang == "en" else letters_ar

    if instruction_lang == "en":
        instruction = (
            "# Instruction\n"
            "Return only the final answer as the option letter inside double square brackets."
        )
        sep = ", "
        or_word = "or"
        valid_prefix = "Valid outputs are only:\n"
    else:
        instruction = (
            "# التعليمات\n"
            "أعد فقط الإجابة النهائية على شكل حرف الخيار داخل أقواس مربعة مزدوجة."
        )
        sep = "، "
        or_word = "أو"
        valid_prefix = "المخرجات المسموح بها فقط هي:\n"

    if num_choices == 2:
        options = f"[[{letters[0]}]] {or_word} [[{letters[1]}]]"
    else:
        options = sep.join(f"[[{letter}]]" for letter in letters[:-1])
        options += f"{sep}{or_word} [[{letters[-1]}]]"

    return f"{instruction}\n{valid_prefix}{options}"


def _format_options(doc):
    return "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(CHOICE_LETTERS, doc["options"])
    )


def add_options_with_text(dataset):
    def _add_field(doc):
        doc["options_with_text"] = _format_options(doc)
        return doc

    return dataset.map(_add_field)


def doc_to_text(doc):
    options = doc.get("options_with_text") or _format_options(doc)
    return (
        f"{build_prompt(instruction_lang='ar', labels_lang='en', num_choices=len(doc['options']))}\n"
        "أجب عن السؤال باستخدام الخيار المناسب من بين A أو B أو C أو D. "
        "الرجاء الرد بالحرف الصحيح فقط دون أي شرح أو معلومات إضافية.\n"
        f"السؤال: {doc['question']}\n"
        f"الجملة: {doc['sentence']}\n"
        "الاختيارات:\n"
        f"{options}\n"
        "الإجابة:"
    )


def doc_to_target(doc):
    answer = doc["answer"]
    if answer not in CHOICE_LETTERS:
        raise ValueError(f"Invalid answer character: {answer!r}")
    return answer
