"""Controles sencillos ajustados exclusivamente con entrenamiento."""

import math
from collections import Counter, defaultdict

from brain.data import Lesson, Prompt
from brain.encoding import EOS, decode_answer, encode_prompt


def require_train(lessons):
    if not lessons or any(x.split != "train" for x in lessons):
        raise ValueError("Las referencias solo pueden ver entrenamiento")


class Majority:
    def __init__(self, lessons: list[Lesson]):
        require_train(lessons)
        self.global_answer = Counter(x.answer for x in lessons).most_common(1)[0][0]
        grouped = defaultdict(Counter)
        for lesson in lessons:
            grouped[lesson.question][lesson.answer] += 1
        self.answers = {q: counts.most_common(1)[0][0] for q, counts in grouped.items()}

    def predict(self, prompts: list[Prompt], max_new_bytes: int = 80) -> list[str]:
        return [self.answers.get(x.question, self.global_answer) for x in prompts]


def grams(prompt: Prompt) -> set[str]:
    text = prompt.situation + "\n" + prompt.question
    return {text[i : i + 3] for i in range(len(text) - 2)}


class NearestExample:
    """Retrieve the nearest taught prompt by character-trigram cosine similarity."""

    def __init__(self, lessons: list[Lesson]):
        require_train(lessons)
        self.answers = [x.answer for x in lessons]
        self.index = defaultdict(list)
        for i, lesson in enumerate(lessons):
            features = grams(lesson.prompt())
            for feature in features:
                self.index[feature].append((i, 1 / math.sqrt(max(1, len(features)))))

    def predict(self, prompts: list[Prompt], max_new_bytes: int = 80) -> list[str]:
        results = []
        for prompt in prompts:
            scores = defaultdict(float)
            for gram in sorted(grams(prompt)):
                for index, weight in self.index[gram]:
                    scores[index] += weight
            best = min(scores, key=lambda i: (-scores[i], i)) if scores else 0
            results.append(self.answers[best])
        return results


class ByteNGram:
    def __init__(self, lessons: list[Lesson], order: int = 5):
        require_train(lessons)
        self.order = order
        self.counts = defaultdict(Counter)
        for lesson in lessons:
            prefix = encode_prompt(lesson.prompt())
            full = prefix + list(lesson.answer.encode("utf-8")) + [EOS]
            for i in range(len(prefix), len(full)):
                for length in range(order):
                    context = tuple(full[i - length : i]) if length else ()
                    self.counts[context][full[i]] += 1

    def predict(self, prompts: list[Prompt], max_new_bytes: int = 80) -> list[str]:
        results = []
        for prompt in prompts:
            history = encode_prompt(prompt)
            output = []
            for _ in range(max_new_bytes):
                for length in reversed(range(self.order)):
                    context = tuple(history[-length:]) if length else ()
                    if context in self.counts:
                        token = self.counts[context].most_common(1)[0][0]
                        break
                output.append(token)
                history.append(token)
                if token == EOS:
                    break
            results.append(decode_answer(output) if EOS in output else decode_answer(output) + "�")
        return results
