# Pianissimo Context Broker

En lokal kontextcontroller för svensk transkribering med **KlangAI/Pianissimo**.
Råtext visas direkt. En valfri liten lokal LLM föreslår begränsade patchar och
ordlistan uppdaterar NeMo-decodern inför nästa ljudsegment.

Det här är en första körbar implementation, inte ett färdigvaliderat journalsystem.

## Kom igång utan modeller

Python 3.11 eller senare. Kärnan använder enbart standardbiblioteket.

```bash
git clone https://github.com/Jtensetti/pianissimo-context-broker.git
cd pianissimo-context-broker
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
pianissimo-context demo
python -m unittest discover -s tests -v
```

Demon fungerar helt utan ASR, GPU eller Ollama. Den visar `raw`, `patch` och
`commit` som JSONL. Exempel: `aj sveden` → `AI Sweden`, med råtexten bevarad.

För din egen inkommande transkription: en rad per **nytt, icke överlappande** segment.

```bash
pianissimo-context --context examples/context.json text
```

Skriv en rad och tryck Enter. Avsluta med EOF (Ctrl+D på Linux/macOS,
Ctrl+Z följt av Enter på Windows). Standardfönstret är 15 sekunder från
segmentets mottagning, mätt med en monoton klocka. En timer låser text också
när ingen ny rad kommer. Råtextens utskrift väntar inte på LLM.

## Lokal LLM

Installera och starta Ollama separat, och ladda en liten lokal instruction-modell.
Ange dess installerade modellnamn:

```bash
pianissimo-context --context examples/context.json --ollama-model DITT_MODELLNAMN text
```

Ingen modell väljs eller laddas automatiskt. Klienten använder
`http://127.0.0.1:11434/api/chat`, JSON-svar, temperatur 0 och 4 sekunders
nätverkstimeout. Endast HTTP på loopback tillåts; proxy och redirects är
avstängda. Modellens första laddning kan ta längre tid än timeouten: värm upp
den före ett livesamtal. Timeout och fel lämnar transkriptionen kvar och ger
ett `controller_error` utan att logga prompt eller felmeddelandets innehåll.
Socket-timeout är inte en hård deadline för hela inferensen. Textens låsning
kontrolleras även när svaret anländer.

## Pianissimo + dynamisk phrase boosting

Installera PyTorch för din hårdvara enligt dess installationsinstruktioner,
och därefter NeMo:

```bash
python -m pip install -e '.[asr]'
pianissimo-context --context examples/context.json --ollama-model DITT_MODELLNAMN audio samtal.wav
```

Ljudfilen måste vara **16 kHz, mono, PCM16 WAV**. Modellvikterna hämtas från
Hugging Face vid första laddningen; själva ASR och LLM-inferensen kör lokalt.
NeMo-installation kan kräva ytterligare systemberoenden och är enklast på Linux.
Inga ljudfiler ingår i repot.

`NemoASR` följer [Klangs dokumenterade decoder-konfiguration](https://huggingface.co/KlangAI/pianissimo-sv#vocabulary):
`greedy_batch`, `boosting_tree.key_phrases_list`, `use_triton=False`, och
`boosting_tree_alpha=0.5` som konservativ startnivå. En ändrad ordlista bygger
om decodern inför nästa segment; en oförändrad lista återanvänds. En tom lista
återställer ursprungskonfigurationen. Om konfigurationen misslyckas försöker
adaptern återställa originalet och rapporterar att boosting inte är tillgänglig.
Återställningsfel stoppar körningen.

CLI:ns audio-läge är en **experimentell filbaserad demonstration**: oberoende
10-sekundersklipp, ingen mikrofon, ingen överlappningshantering och ingen
cache-aware streaming. Ord vid klippgränser kan tappas. LLM steget är sekventiellt
mellan klippen, medan råtext skickas före det steget. Det är inte ett latensmått
för live-ASR. `--chunk-seconds` väljer klipplängd.

Klang beskriver Pianissimo som en svensk finjustering av Parakeet **v3**, inte v2.
Alternativ modell kan anges med `--asr-model`, men
`nvidia/parakeet-tdt-0.6b-v2` är en engelsk modell och är inte svensk standard här.

## Kontext och tillåtna ändringar

`examples/context.json` visar `topic`, `people`, `organisations`, `terms`,
`facts` och `aliases`. Fakta är användarens uppgifter; LLM får inte skapa nya
fakta eller roller. Fakta används som kontext, aldrig som transkriptinnehåll.

* Automatiska semantiska namnbyten kräver ett **uttryckligt alias**. Att
  `Socialstyrelsen` står i ordlistan tillåter inte att `socialen` byts ut.
* Formatpatchar får ändra kapitalisering, skiljetecken och blanksteg, men
  måste bevara bokstavs-/sifferföljden. Tal inklusive decimaltecken och kända
  svenska negationer skyddas separat. Skiljetecken och sammanskrivning kan
  fortfarande påverka tolkningen: valideringen är en konservativ regel,
  inte ett bevis på oförändrad betydelse.
* Patchar anger `start`, `end`, `source`, `replacement`, `confidence`, `reason`.
  Gränser är Python Unicode-teckenindex, slut exklusivt. En JS-klient bör
  använda eventets kompletta `text`, eftersom JS räknar UTF-16-kodenheter.
* Exakt källtext, hela ordgränser, rätt revision och olåst segment krävs.
  Överlappande patchar avvisas som en grupp. Tröskel är 0.95, men LLM:s
  confidence är **inte** en kalibrerad sannolikhet eller ASR-confidence.
* Dynamiska termer måste förekomma som hela ord/fraser i råtexten. Reparations-
  output används aldrig som belägg för nya termer. Rå-ASR kan fortfarande
  innehålla fel; håll boost låg och utvärdera på egna inspelningar.
* Högst 100 boostfraser; explicita termer prioriteras, dynamiska termer hålls
  i en begränsad lista med nya termer först. Prompten har högst 4 000 tecken
  föregående råtext plus aktuellt segment och strukturerad kontext.

## Koppla till din app

```python
import asyncio
from pianissimo_context import Broker, Context, Ollama

async def main():
    broker = Broker(
        Context(organisations=("AI Sweden",), aliases={"aj sveden": "AI Sweden"}),
        controller=Ollama("DITT_MODELLNAMN"),
        emit=lambda event: print(event),  # Ersätt med en snabb UI/eventkö.
        mutable_seconds=15,
    )
    segment = broker.add("vi pratade med aj sveden")  # raw event synkront
    await broker.repair(segment.id)                  # kör i din bakgrundsarbetare
    broker.finish()                                 # commit alla återstående

asyncio.run(main())
```

En broker och en dedikerad ASR-modellinstans per samtal. Kör broker-metoder på
samma asyncio-loop. Schemalägg `tick()` ungefär var 100 ms. Använd en enda
bakgrundsarbetare som försöker reparera nya olåsta segment; samtidiga
`repair()`-anrop medan modellen arbetar returnerar `False`. Text-CLI har
en sådan arbetare och en begränsad notifieringskö. Segment som hinner låsas
behåller sin råtext. Efter att klienten sparat commit-event kan
`release_committed()` frigöra äldre transkript ur minnet.

| Event | Klientens åtgärd |
| --- | --- |
| `raw` | Visa råtext och spara segmentets ID/revision |
| `patch` | Kontrollera `base_revision`, ersätt segmenttext, uppdatera revision |
| `commit` | Lås segmentet och spara sluttext |
| `glossary` | Uppdatera ASR-ordlistan inför nästa decoding |
| `controller_error` | Behåll befintlig text |
| `asr_warning` | Informera att grunddecodern används |

Vid direkt ASR-integration: `asr.update_glossary(broker.glossary())` före nästa
`asr.transcribe(path)`. Adaptern låser config-/inferenzanrop så de inte överlappar.
Detta är uppdatering mellan anrop, **inte** NeMo:s per-stream boosting-API.

## Verifierat och nästa steg

Enhetstester täcker råtext först, alias, commit under pågående LLM-anrop,
felaktiga offset/revisioner, överlapp, numeriska ändringar, negationer,
malformade modellförslag, råtextgrundad termextraktion och felhantering.
CI kör kärnan på Python 3.11–3.13. NeMo-konfiguration testas med en modelldubbel.

**Inte verifierat ännu:** verklig ASR på GPU/CPU, kvalitet med en lokal LLM,
förbättrad WER eller faktisk live-latens. Inga modellvikter eller ljud har
laddats för de testerna. Nästa steg är ett anonymiserat ljudtest med baseline,
boosting och broker, inklusive kontroll av felaktigt infogade namn.

Inte implementerat i v0.1: akustisk omprövning av osäkra delsegment,
ordnivå-confidence från ASR, diarisation, mikrofon/UI, eller NeMo:s per-stream
streamingintegration. De får inte simuleras med LLM:s egna confidencevärden.

## Modellkällor och attribution

* [KlangAI/pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv) — Klang AI AB,
  2026, modelllicens CC BY 4.0. Projektet distribuerar inga modellvikter.
* [NVIDIA NeMo phrase boosting](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_customization/word_boosting.html).
* [Parakeet TDT v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2).

Ingen inbyggd molninferens eller telemetri. JSONL-output innehåller transkript:
lagring och åtkomst till eventen bestäms av din app.
