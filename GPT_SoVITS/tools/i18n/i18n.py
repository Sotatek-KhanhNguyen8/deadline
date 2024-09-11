import json
import locale
import os


def load_language_list(language):
    current_dir = os.path.dirname(__file__)
    i18n_file_path = os.path.join(current_dir, "locale", f"{language}.json")
    with open(i18n_file_path, "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        current_dir = os.path.dirname(__file__)
        i18n_file_path = os.path.join(current_dir, "locale", f"{language}.json")
        if not os.path.exists(i18n_file_path):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
