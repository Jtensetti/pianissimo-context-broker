# Pianissimo Context Broker

En lokal kontextcontroller för **live-transkribering med Pianissimo**. Ge en
kort beskrivning av samtalet, låt Pianissimo leverera råtext direkt och låt en
liten lokal språkmodell föreslå rättningar och följa hur ämnet utvecklas.

Version 0.2 innehåller fri initial kontext, kontextbedömning var 25:e sekund,
begränsat samtalsminne, konservativa patchar, JSONL-events och valfri
mikrofoninmatning. Det är en körbar prototyp; verklig modellkvalitet och
hårdvarulatens är ännu inte uppmätta.

## Så följer brokern samtalet

Tre delar av kontexten hålls separat:

1. **Initial bakgrund** – användarens beskrivning finns kvar hela samtalet.
2. **Aktuell kontext** – språkmodellens preliminära ämnesbild omprövas var
   20–30:e sekund när det finns nytt tal. Standard är 25 sekunder, med upp
   till 30 sekunders råtext som underlag.
3. **Samtalsminne** – upp till åtta tidigare ämnen, korta sammanfattningar och
   belägg bevaras när ämnet byts. Modellen kan återknyta till en tidigare
   diskussion utan att hela transkriptet skickas med varje gång.

Kontextens sammanfattningar är **hypoteser**, inte ett register över verifierade
fakta. Nya roller, identiteter och sakuppgifter ska inte härledas till journalen.
Initial bakgrund och tidigare ämnen kan hjälpa modellen tolka ett uttryck;
aktuell råtext har företräde. En utvikning om vädret ska få vara en utvikning.

Modellen granskar nya olåsta segment i en separat arbetare. Kontext och
ASR-ordlista har olika livslängd: tidigare ämnen ligger kvar i minnet men deras
inlärda boostfraser ersätts vid nästa kontextbedömning. Explicit konfigurerade
fraser i en JSON-kontext ligger kvar tills appen ändrar dem.

## Installation och demo

Python 3.11 eller senare. Kärnan har inga externa körberoenden.

```bash
git clone -b feat/local-context-broker https://github.com/Jtensetti/pianissimo-context-broker.git
cd pianissimo-context-broker
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
pianissimo-context demo
python -m unittest discover -s tests -v
```

Demon kräver ingen modell och visar råtext, aliaspatchar och låsning. Kommando-
flaggan `--context examples/context.json` ger ett strukturerat komplement till
fri kontext: topic, people, organisations, terms, facts och aliases.

## Live från din befintliga ASR-app

Installera och starta Ollama separat, och ladda den lilla instruction-modell du
vill använda. Ange dess installerade modellnamn; inget namn hårdkodas eller
laddas automatiskt.

```bash
pianissimo-context --ollama-model DITT_MODELLNAMN --initial-context "Detta är en intervju med Jonatan Permert från AI Sweden och Niklas Silfverström från Klang. De diskuterar hur Pianissimo kan inkluderas i Svea, en AI-chattbot för offentlig sektor." live
```

Skicka en rad per **nytt, icke överlappande ASR-segment** till stdin. `text` är
ett alias för `live`. stdout ger JSONL-events. Råtexten väntar inte på LLM.
Avsluta med EOF. `--review-seconds 25` anger kontextkadens.

* Vid start extraherar LLM aktuellt ämne och relevanta fraser ur initialtexten.
* Under samtalet granskas varje nytt segment med initial bakgrund, aktuell
  kontext, tidigare ämnen och begränsad föregående råtext.
* På kontextkadensen analyseras råtexten på nytt. En ny ämnesbild kräver korta
  belägg som bokstavligen finns i det råa underlaget. Nya boosttermer måste
  förekomma i dessa belägg, inte i språkmodellens egna rättningar.
* Kontext uppdateras även när segment från underlaget redan låsts och frigjorts.
* En ny kontextrevision gör att fortfarande olåsta segment kan granskas igen.
* Vid tystnad låses texten, men modellen gör inte upprepade tomma ämnesbedömningar.

Standard är ett **15 sekunders muterbart fönster från textens mottagning**, mätt
med en monoton klocka. Därefter blir segmentet permanent. Detta fönster är
separat från kontextkadensen: en bedömning efter 25 sekunder öppnar inte text
som redan låsts. `--mutable-seconds` ändrar låsningstiden.

## Live från mikrofon

Installera PyTorch för din hårdvara och därefter de valfria beroendena:

```bash
python -m pip install -e '.[asr,live]'
pianissimo-context --ollama-model DITT_MODELLNAMN --initial-context "Intervju om Pianissimo, Svea och transkribering i offentlig sektor." microphone
```

NeMo-modellen hämtas första gången före mikrofonstart. Mikrofonen öppnas som
16 kHz mono PCM16. PortAudio måste finnas på systemet; vissa mikrofoner saknar
stöd för vald samplerate. `--device` väljer inmatningsenhetens namn.

Live-ASR kör **oberoende ljudklipp**, som standard fyra sekunder; ändra med
`microphone --chunk-seconds 3`. Mikrofonen fortsätter fånga ljud medan ASR och
LLM arbetar. Ljudkön rymmer tre klipp. Om ASR halkar efter tappas nya klipp med
ett uttryckligt `audio_drop`-event och tidsintervall. Det finns ingen tyst,
obegränsad kö som bygger upp fördröjning. Ctrl+C stoppar inspelningen och låser
redan transkriberad text; köade och ofullständiga klipp transkriberas inte då.

Detta är chunkad live-ASR, **inte cache-aware neural streaming**. Klippgränser
kan dela ord; överlappningssammanfogning, diarisation och ordnivå-confidence är
inte implementerade. Råtext kommer efter klippets ASR; språkmodellens rättningar
kommer därefter utan att blockera nästa mikrofonklipp. Den praktiska fördröjningen
beror på klipplängd, hårdvara och modeller.

## Vad får automatiskt rättas?

* Explicit angivna alias: exempelvis `aj sveden` → `AI Sweden`.
* Kapitalisering, interpunktion och blanksteg med bibehållen bokstavs-/sifferföljd.
* I live-läge: små felskrivningar av **aktuella domäntermer och sammansättningar**,
  inte bara namn. Ersättningen måste finnas i aktiv ordlista; källan måste vara
  6–60 tecken, högst fyra ord, med hög stränglikhet och högst två teckens
  ändringsbudget. Exempel: `transkriberingsmodulen` → `transkriberingsmodellen`
  när den senare är en aktiv term.

Exakt källtext, rätt offset/revision och olåst segment krävs. Tal inklusive
tecken i decimaler, kända svenska negationer och vid termsubstitution vissa
svenska räkneord skyddas. Överlappande batchar avvisas. LLM-confidence måste
vara minst 0.95, men det är en **heuristik, ingen akustisk sannolikhet**.
Stränglikhet är inte heller ett bevis för vad som sades: nära ord kan ha olika
betydelse. Validera automatisk termrättning på egna inspelningar.

Större ordbyten som LLM tror kan vara transkriptionsfel blir ett `suggestion`
med `text_changed=false`. De skriver inte om den visade transkriptionen.
`--suggestions-only` stänger av automatiska nära termbyten; uttryckliga alias
och formateringspatchar är fortfarande aktiva. Ett ord får aldrig ersättas
bara för att meningen verkar främmande för ämnet. Ingen akustisk verifiering
eller resolver av osäkra ljuddelar finns ännu.

## Integration i Python

```python
from pianissimo_context import AdaptiveBroker, LiveSession, Ollama

async def consume(raw_segments):
    broker = AdaptiveBroker(
        initial_context="Intervju om Pianissimo i Svea, en AI-chattbot för offentlig sektor.",
        controller=Ollama("DITT_MODELLNAMN"),
        emit=events_to_ui_queue,  # Din snabba, synkrona eventkö.
        review_seconds=25,
    )
    async with LiveSession(broker) as session:
        async for raw_text in raw_segments:
            session.push(raw_text)  # raw event direkt; ingen LLM-await här
```

`LiveSession` sköter timer, seriell modellgranskning, kontextbedömningar,
begränsade notifieringar och stängning. En session/broker per samtal, alla
broker-anrop på samma asyncio-loop. Callbacken får inte blockera. Committed
segment frigörs från RAM, så klienten måste spara dem genom commit-eventen.
Sammanfattningsminnet är begränsat till åtta ämnen och råhistoriken till
30 sekunder, max 200 segment/8 000 tecken i kontextunderlaget.

Den enklare `Broker` behåller det tidigare API:t och tillåter endast alias och
formateringspatchar som standard. Den har ingen automatisk kontextkadens.

| Event | Klientens åtgärd |
| --- | --- |
| `raw` | Visa råtext, spara ID/revision och original |
| `patch` | Matcha base_revision, ersätt segmentets text och revision |
| `commit` | Lås och spara segmentets sluttext |
| `context` | Visa eventuell ämnesbild; diagnostik, inte journalfakta |
| `glossary` | Uppdatera ASR-ordlista inför nästa decoding |
| `suggestion` | Visa ett osäkert förslag separat; ändra inte texten |
| `controller_error` | Behåll text och tidigare kontext |
| `audio_ready` | Mikrofonflödet förberett |
| `audio_warning`, `audio_drop`, `asr_warning` | Visa aktuell ljud-/ASR-begränsning |

Patchoffset är Python Unicode-teckenindex, slut exklusivt. JS-klienter bör
använda eventets kompletta `text`, eftersom JS räknar UTF-16-kodenheter.

## ASR-ordlista och filtest

`NemoASR` följer [Klangs decoder-exempel](https://huggingface.co/KlangAI/pianissimo-sv#vocabulary):
`greedy_batch`, `boosting_tree.key_phrases_list`, `use_triton=False` och
`boosting_tree_alpha=0.5`. Ändrad ordlista bygger om decodern inför nästa klipp;
oförändrad lista återanvänds. Dedikerad modellinstans per samtal; lås skyddar
konfiguration och transkribering från samtidiga anrop. En tom lista återställer
originalet. Boosting-fel återställer grunddecodern och rapporteras; om även
återställningen misslyckas stoppas körningen.

```bash
pianissimo-context --context examples/context.json audio samtal.wav
```

Filtestet kräver en 16 kHz mono PCM16 WAV och behåller statisk kontext.
**Den adaptiva kontextloopen används i live/text och microphone, inte i audio.**
Pianissimo är Klangs svenska finjustering av Parakeet v3. Den alternativa
`nvidia/parakeet-tdt-0.6b-v2` är engelsk och används inte som svensk standard.

## Lokalitet och verifiering

Ollama använder endast HTTP-loopback, utan proxy eller redirects. Modellen
väljs av användaren. Första modellstarten kan ta tid; värm upp den före samtalet.
Socket-timeout är fyra sekunder, inte en hård tidsgräns för hela inferensen.
Försenade patchar kan inte ändra låst text, och kontextbedömningar äldre än en
hel kontextperiod avvisas. Vid fel behålls föregående text/kontext. Undantagens
meddelanden och promptinnehåll loggas inte.

Ingen inbyggd molninferens eller telemetri. Modellvikter hämtas vid första
installation/laddning; inferens sker lokalt. JSONL-events innehåller transkript
och kontext: din app bestämmer lagring och åtkomst.

Enhetstester täcker patchskydd, råtext först, låsning under LLM-anrop, decoder-
återställning, initialtext, kadens, råtextbelägg, ämnesminne, termers livslängd,
ämnesbyte, tystnad och parallell råtextinmatning under kontextanalys.
CI kör kärnan på Python 3.11–3.13. NeMo och språkmodeller testas med dubblar.
**Verklig mikrofonhårdvara, modellinferens, WER-förbättring och live-latens är
inte verifierade i utvecklingsmiljön.**

Modellattribution: [KlangAI/pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv),
Klang AI AB, 2026, CC BY 4.0. Inga modellvikter distribueras av projektet.
Se även [NVIDIA NeMo phrase boosting](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_customization/word_boosting.html).
