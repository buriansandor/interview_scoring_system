# Kétlépcsős interjúpontozás

Az elsődleges modell minden választ először értékel. A második lépés
technikai hiba vagy alacsony confidence miatt indulhat; ez két külön
mechanizmus, külön okkal és mentett státusszal.

## Konfiguráció

```json
{
  "primary": {"provider": "groq_free", "model": "openai/gpt-oss-120b"},
  "secondary": {"provider": "gemini", "model": "gemini-2.5-flash"},
  "review": {
    "confidence_threshold": 0.5,
    "primary_weight": 0.6,
    "secondary_weight": 0.4,
    "disagreement_threshold": 0.25
  },
  "input_csv": "tests/results/test.csv",
  "rules_json": "evaluation.json",
  "output_csv": "scored_answers.csv"
}
```

A példában szereplő küszöbök és súlyok szerkeszthető kiindulóértékek,
nem validált tudományos paraméterek. A szállított alapkonfiguráció
elsődleges helyi modellt és `secondary: null` beállítást használ.

- **`primary`:** szolgáltatót tartalmaz; a további szerepszintű mezők
  felülírják a `providers` profil alapértékeit.
- **`secondary`:** opcionális szerep saját szolgáltatóval és modellnévvel.
  Azonos szolgáltatón két külön modell is beállítható.
- **`secondary: null`:** szükség esetén az elsődleges modell kap egy új,
  külön felülvizsgálati kérést. Nem kapcsolja ki a második lépést.
- **Régi `provider`:** továbbra is támogatott elsődleges kiválasztásként,
  de nem szerepelhet egyszerre a `primary` mezővel.

A szerepek megengedett felülírásai: `model`, `api_key_env`, `api_key_file`,
`json_mode`, `requests_per_minute`; helyi szerepnél `base_url` is.
Eltérő kulcsfájlokhoz eltérő `api_key_env` neveket használj.

## A két mechanizmus

- **Technikai fallback:** az elsődleges lépés a megengedett próbálkozások
  után sem ad érvényes eredményt. A második lépés oka `technical_failure`;
  nincs érvényes első pontszám, ezért a második önállóan értékel.
- **Bizonytalansági felülvizsgálat:** az első eredmény érvényes, de a
  confidence szigorúan a küszöb alatt van. Az ok `low_confidence`; a
  második megkapja az érvényes első értékelést is.

Pontosan egy elsődleges és legfeljebb egy második értékelési lépés van.
Mindkét lépésen belül a `settings.max_attempts` korlátozza a technikai
és kimenetjavító próbálkozásokat. Nincs harmadik felülvizsgálat, és nincs
olyan ciklus, amely addig fut, amíg a confidence mesterségesen megnő.
A régi `settings.max_reflections` kompatibilitásból elfogadott, de a
kétlépcsős útvonalon nem vezérel újabb ismétléseket.

A második modell megkapja az eredeti kérdést, a teljes rubrikát,
a minimumot és maximumot, a válasz szövegét, a második lépés okát és az
első érvényes értékelést, ha létezik. Hibás első válaszból nem képezünk
ál-pontszámot, és nem adjuk tovább érvényesként.

A prompt kizárólag az első írott indoklásának és a megadott bizonyítékoknak
az értékelését kéri. Tiltja a rejtett gondolatmenet vagy belső döntési
folyamat ismeretének állítását. A második confidence alacsony is maradhat.

## Mentési döntések

| Elsődleges | Második lépés | Mentett eredmény |
|---|---|---|
| Érvényes, confidence legalább a küszöb | Nem indul | Elsődleges pontszám |
| Érvényes, alacsony confidence | Érvényes | Konfigurált súlyozott átlag |
| Érvényes, alacsony confidence | Sikertelen | Elsődleges pontszám, kötelező felülvizsgálati jelzés |
| Sikertelen | Érvényes | Kizárólag a második pontszám |
| Sikertelen | Sikertelen | Üres végső pontszám, hibastátusz |
| Két érvényes, jelentősen eltérő pontszám | Érvényes | Átlag megmarad, kötelező felülvizsgálati jelzés |

A confidence küszöbbel való egyenlősége nem indít második lépést.
A második modell saját alacsony confidence-e is felülvizsgálati jelzést
eredményez. Sikeres, megfelelő confidence-ű második értékelés és
küszöb alatti eltérés esetén a felülvizsgálat befejezettnek jelölhető;
az első alacsony confidence ekkor is változatlanul megmarad a CSV-ben.

## Súlyozás és eltérés

Csak két érvényes pontszám esetén:

\[
s_{\mathrm{final}} =
\frac{w_p s_p + w_s s_s}{w_p + w_s}
\]

A súlyok véges, nemnegatív számok; legalább az egyik pozitív. Nem kell
1-re összeadódniuk: a program normalizálja őket. Nem kerekít automatikusan,
és egyetlen érvényes értéknél nem helyettesíti nullával a hiányzó másikat.

A normalizált eltérés:

\[
d = \frac{|s_p-s_s|}{\mathrm{range\_max}-\mathrm{range\_min}}
\]

A `d` csak két érvényes pontszámnál létezik. A skála végpontjainak
véges számoknak kell lenniük, és `range_max > range_min` szükséges.
Azonos vagy fordított végpontoknál a program LLM-hívás előtt hibát jelez.

Ha `d >= disagreement_threshold`, `needs_review=True`, az átlag pedig
megmarad. Példa: 1–5 skálán 2 és 4 pont eltérése `d=0.5`;
0.6/0.4 súlyokkal a végső pontszám 2.8.

## CSV és visszakövethetőség

A korábbi `score`, `confidence`, `reasoning` mezők mellett mentjük:

- **Első és második értékelés:** `primary_score`, `secondary_score`,
  a két confidence és reasoning, konfigurált szolgáltató/modell,
  próbálkozásszám és lépésstátusz.
- **Döntés:** `second_step_reason`, `secondary_mode`, `status`,
  `aggregation_method`, `needs_review`, `review_reasons`, `failure_reason`.
- **Számítás:** konfigurált és tényleges normalizált súlyok,
  `absolute_disagreement`, `d`, skálavégpontok és küszöbök.
- **Futási állapot:** `stop_requested`, közös kéréskeret kimerülésekor.

Súlyozott átlaghoz a végső `confidence` üres, mert egyik modell sem
adott confidence-et magára az aggregált értékre. A két eredeti confidence
külön megmarad; `confidence_source=not_defined_for_weighted_score`.
Egyetlen megtartott értékelésnél annak saját confidence-e szerepel.

A sikertelen értékelés is kap auditálható sort; az üres pontszám nem 0.
Az azonos modelles újraértékelés `secondary_mode=same_model_review`:
nem állítjuk, hogy két független modell véleményéről van szó.

## Kulcsok, keretek és notebookok

A CLI az elsődleges után, csak szükség esetén inicializálja a konfigurált
másodlagost. Hiányzó másodlagos kulcs nem akadályozza az elsődleges
értékelést: ha szükség lenne a másodikra, annak hibáját külön mentjük.
Érvénytelen alapkonfiguráció továbbra is indulási hiba.

A notebookok ugyanazt az `app.py`-t futtatják. A Colab-wrapper a
kifejezetten konfigurált másodlagos Secrets-kulcsát előre előkészítheti a
gyermekfolyamat számára; hiba esetén figyelmeztet, de az elsődleges
futtatását nem állítja le. Ez még nem LLM-hívás.

A `max_requests_per_run` közös keret: mindkét modell kérései és retry-jai
beleszámítanak. Kimerülésekor nincs új hívás, a már meglévő érvényes
elsődleges érték és a státusz megmarad, a program részleges mentés után
2-es kilépési kódot ad.

Alapértelmezésben nincs másik szolgáltatóra váltás. Ha viszont helyi
elsődleges mellé felhős másodlagost állítasz, az kiváltott második
lépésben megkapja a választ és az első érvényes értékelést. Ezt az
adatkezelésnél és a díjcsomagnál is vedd figyelembe.

Kutatási használat előtt külön validáld a súlyokat, küszöböket és azt,
hogy a skáládon indokolt-e átlagot számolni. A rendszer a confidence-et
nem kezeli kalibrált valószínűségként, és a `d` sem ilyen valószínűség.
