import pyttsx3

def main():
    engine = pyttsx3.init()  # on Windows this is SAPI5
    voices = engine.getProperty("voices") or []
    print(f"Found {len(voices)} voices:\n")

    for i, v in enumerate(voices):
        name = getattr(v, "name", "")
        vid = getattr(v, "id", "")
        langs = getattr(v, "languages", None)
        gender = getattr(v, "gender", None)
        age = getattr(v, "age", None)

        print(f"[{i}] name={name!r}")
        print(f"    id={vid!r}")
        print(f"    languages={langs!r} gender={gender!r} age={age!r}\n")

if __name__ == "__main__":
    main()