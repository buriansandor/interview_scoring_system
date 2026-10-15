# Interjúpontozó: szolgáltatóválasztás JSON-konfigurációval

A szolgáltatót az `llm_config.json` fájlban választhatod ki. A README az
útmutató, nem végrehajtható konfiguráció: így egy dokumentációs módosítás
nem változtatja meg véletlenül a modellkapcsolatot.

## Szolgáltató kiválasztása

A teljes példakonfiguráció szerepel a csomagban. Általában csak ezt az egy
mezőt kell átírnod:

```json
{
  "provider": "groq_free"
}
```

Ez önálló minimális konfigurációként is érvényes. A többi mezőhöz a program
alapértékeket használ; a minimális változatnál a bemeneti fájlok automatikus
keresése csak akkor működik, ha egyértelmű a találat.

| `provider` | Alapértelmezett modell | API-kulcs fájlja |
|---|---|---|
| `local` | `interview-scorer` | Nem szükséges |
| `gemini` | `gemini-2.5-flash` | `gemini.key` |
| `mistral_free` | `mistral-small-latest` | `mistral.key` |
| `groq_free` | `openai/gpt-oss-120b` | `groq.key` |
| `openrouter` | `openrouter/free` | `openrouter.key` |

Az alapérték modellválasztás, nem minőségi garancia. A fiókban elérhető
modellek és a szolgáltatói kvóták változhatnak. A Gemini példa a
[publikált ingyenes modellkínálatból](https://ai.google.dev/gemini-api/docs/pricing)
származik; az OpenRouter `openrouter/free` az
[ingyenes modellrouter](https://openrouter.ai/docs/guides/routing/routers/free-router).

## Helyi szerver

Egy minimális, de konkrét fájlokat megadó beállítás:

```json
{
  "provider": "local",
  "input_csv": "tests/results/test.csv",
  "rules_json": "evaluation.json",
  "output_csv": "scored_answers.csv",
  "providers": {
    "local": {
      "base_url": "http://127.0.0.1:8080/v1",
      "model": "interview-scorer",
      "json_mode": "json_object"
    }
  }
}
```

llama.cpp indítása egy már letöltött chat/instruct GGUF-modellel:

```shell
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080 --alias interview-scorer
```

A szerver OpenAI-kompatibilis chat-completions végpontját és modellaliasait
a [llama.cpp dokumentáció](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
ismerteti. Ollamánál a `base_url` legyen `http://127.0.0.1:11434/v1`, a
modellnév pedig a telepített modell pontos tagje, az
[Ollama kompatibilitási leírása](https://docs.ollama.com/api/openai-compatibility)
szerint.

A program nem indítja el a modellszervert és nem tölt le modellt. Helyi
módban nem kér felhős kulcsot és nem vált át felhős szolgáltatóra.
Ha a helyi szerver hitelesítést használ, külön `api_key_file` vagy
`api_key_env` beállítható. A végpontot csak megbízható gépre irányítsd.

## Felhős kulcsok

A kiválasztott szolgáltató `.key` fájljában kizárólag az API-kulcs legyen,
idézőjelek és változónév nélkül. A fájlok Git-ignore alatt vannak; ne
oszd meg és ne töltsd fel őket a repóba.

Környezeti változó is használható: `GEMINI_API_KEY`, `MISTRAL_API_KEY`,
`GROQ_API_KEY`, `OPENROUTER_API_KEY`. Ezek elsőbbséget élveznek a kulcsfájllal
szemben. A program kizárólag a kiválasztott szolgáltató kulcsát keresi.
Az inline `api_key` mező a JSON-ban szándékosan nem engedélyezett.

A JSON-szolgáltatóválasztás elsőbbséget élvez a régi `LLM_PROVIDER`
környezeti változóval szemben. A korábbi környezeti változós helyi indítás
csak akkor érvényesül, ha nincs konfigurációs fájl.

## Indítás és fájlok

A repó gyökérkönyvtárából:

```shell
pip install -r requirements.txt
python app.py
```

Másik konfigurációval:

```shell
python app.py --config sajat_beallitas.json
```

A bemeneti, kimeneti és kulcsfájlok relatív útvonalai a konfigurációs fájl
mappájához képest értendők. A `specs/context.md` és a
`specs/specification.md` továbbra is az indítási könyvtárhoz képest
töltődik, ezért a programot a repó gyökeréből indítsd.

A szállított konfiguráció a repó jelenlegi `tests/results/test.csv`
mintáját használja. Valós feldolgozás előtt állítsd át az `input_csv`
értékét a saját CSV-dre.

A `rules_json` a pontozási szabályfájl, nem az LLM-konfiguráció.
Automatikus fájlkeresésnél a program kihagyja a konfigurációt és a kimeneti
CSV-t. Több lehetséges bemenet esetén megáll, nem találomra választ.

## Notebookok: helyi Jupyter és Colab

A repó mindkét notebookja ugyanazt a JSON-t és `app.py`-t használja.
A `notebook_support.py` segédmodul futtatja a programot, majd visszaadja
a kimeneti CSV-t DataFrame-ként. Nincs külön, Geminihez kötött pontozókód.

- **Helyi notebook:** `interview_scoring_system.ipynb`. A repó gyökeréből
  nyisd meg, állítsd be a JSON-t, majd futtasd a cellákat. Másik
  konfigurációhoz a `CONFIG_PATH` útvonalat változtasd.
- **Colab:** `Interview_answer_scoring_system-colab_version.ipynb`.
  A setup csak hiányzó checkout esetén klónoz; meglévő fájlokat nem
  állít vissza és nem vált ágat. A PR nyitva tartása alatt az alapértelmezett
  `REPO_REF` a `feat/json-llm-provider-config`; merge után válts `main`-re.
- **Secrets:** kizárólag a kiválasztott felhős szolgáltató hiányzó kulcsát
  keresi, a profil `api_key_env` nevével. A környezeti változó és a
  kiválasztott `.key` fájl elsőbbséget élvez. Helyi módhoz nem kér Colab
  Secretet. A kulcs a gyermekfolyamat környezetébe kerül, nem parancssori
  argumentumba, notebookforrásba vagy automatikusan létrehozott fájlba.
- **Részleges eredmény:** a futási keret kimerülését figyelmeztetés és
  `result_df.attrs["partial_results"] = True` jelzi. Egyéb futási hibánál
  nem mutatja korábbi futás elavult CSV-jét.

A notebook nem nyomtatja ki automatikusan a gyermekfolyamat teljes naplóját.
Az `agent.log` viszont tartalmazhat interjúszöveget, a DataFrame előnézete
pedig válaszokat: megosztás előtt töröld a notebook kimeneteit, és a naplót
is kezeld bizalmasan. A korábbi Colab-cellák már nem írják át az `app.py`-t
és nem telepítik a szükségtelen `guardrails-ai` csomagot.

Hosztolt Colabban a `localhost` a távoli futtatókörnyezet, nem az otthoni
géped. Otthoni llama.cpp/Ollama szerverhez használd a helyi notebookot vagy
helyi runtime-ot; ne nyiss hitelesítés nélküli publikus szervert.
A külső Kaggle-másolatot ez a repómódosítás nem szinkronizálja.

## Híváskorlátok és ellenőrzés

```json
{
  "settings": {
    "timeout": 120,
    "max_tokens": 1024,
    "max_attempts": 3,
    "max_reflections": 1,
    "temperature": 0.1,
    "max_requests_per_run": 50
  }
}
```

- `timeout`: egy kérés időkorlátja másodpercben.
- `max_tokens`: egy válasz maximális kimeneti tokenkerete.
- `max_attempts`: egy interjúválasz összes modellhívása, javításokkal együtt.
- `max_reflections`: alacsony confidence esetén megengedett újraértékelések.
- `max_requests_per_run`: az egész futás kéréskerete, újrapróbálkozásokkal együtt.
- `requests_per_minute`: szolgáltatói profilban megadott ütemezési korlát.

A futási kéréskeret kimerülésekor a program elmenti az addigi eredményeket
és 2-es kilépési kóddal megáll. Új indítás új futási keretet jelent:
ez nem napi kvótanyilvántartás, tokenkeret vagy szolgáltatói költési limit.
A Gemini SDK belső automatikus retry-ja ki van kapcsolva, hogy az
alkalmazás kezelje a próbálkozásokat.

Minden szolgáltatónál kötelező a pontszám tartományának, a numerikus
típusoknak, a confidence 0 és 1 közötti értékének és az indoklásnak az
ellenőrzése. A program nem emeli mesterségesen a confidence értékét.
Kimerült javítási keret után a hibás eredményt nem menti pontozásként.

Ha a szerver nem fogadja a `response_format` opciót, a profilban állítsd
a `json_mode` értékét `none`-ra. A Python-oldali ellenőrzés ettől nem
kapcsolódik ki. Gemini esetén az SDK JSON-kimeneti beállításait használjuk.

## Mit jelent az ingyenes mód?

**A `groq_free` és `mistral_free` név nem tudja átállítani a szolgáltatói
fiókod számlázását.** A kliens nem tud általános garanciát adni a nulla
költségre, ha fizetős fiókhoz vagy számlázásra jogosult projekthez tartozó
kulcsot adsz meg.

- Mistralnál maradj Free mode-ban, a pay-as-you-go legyen kikapcsolva a
  [szolgáltatói előfizetési beállításokban](https://docs.mistral.ai/admin/billing-usage/subscriptions).
- Groqnál Free Planhez tartozó szervezet kulcsát használd, és ellenőrizd
  az aktuális [modellkvótákat](https://console.groq.com/docs/rate-limits).
- Gemininél ingyenes projektet és az ingyenes csomagban elérhető modellt
  válassz a [díjszabás alapján](https://ai.google.dev/gemini-api/docs/pricing).
- OpenRouternél a program csak `openrouter/free` vagy `:free` végű
  modellazonosítót enged, nulla bemeneti és kimeneti tokenárat ír elő,
  és tiltja a provider fallbacket az
  [API routing paraméterei szerint](https://openrouter.ai/docs/guides/routing/provider-selection).
- Helyi modellnél nincs szolgáltatói API-díj, de a hardver és az áram nem ingyenes.

Nincs automatikus szolgáltatóváltás. Elérhetőségi hiba vagy kvótakimerülés
nem kapcsol be egy másik, esetleg fizetős szolgáltatót.

## Tesztelés és korlátok

```shell
python -m unittest discover -s tests -v
```

A tesztek konfigurációt, kulcsválasztást, validálást, híváskeretet,
ütemezést, HTTP-kéréseket és a teljes CSV-feldolgozást vizsgálnak.
A 62 teszt tiszta virtuális környezetben is sikeresen lefutott, és a
függőségellenőrzés nem talált hibás követelményt.
A HTTP-tesztek helyi tesztszervert, a Gemini-teszt mockolt SDK-t használ:
nem minősítik a valós modellek pontozási minőségét vagy aktuális
elérhetőségét. Éles, felhős hívást és valódi helyi inferenciát nem futtattunk.

A notebookok kódcelláit is ellenőrizzük, és mindkét notebook feldolgozási
folyamata lefut helyi HTTP-tesztszerverrel. A csomagtelepítő és klónozó
setupcellákat ebben a tesztben kihagyjuk; valódi Colab-felületet nem
vezéreltünk. A Colab Secrets működését mockolt hozzáféréssel teszteljük.
Éles kutatási adatok előtt anonimizálj, ellenőrizd a szolgáltató
adatkezelését, és hasonlítsd a pontozást kézzel értékelt mintához.
