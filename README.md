# Pianissimo Context Broker

En lokal kontextcontroller för **live-transkribering med Pianissimo**. Ge en
kort beskrivning av samtalet, låt Pianissimo transkribera och låt en liten lokal språkmodell granska
texten före visning, rätta hörfel och följa hur ämnet utvecklas.

Version 0.4 innehåller fri initial kontext, kontextbedömning var 25:e sekund,
redigerbart samtalsminne, lokala kontextuella patchar, JSONL-events och valfri
mikrofoninmatning. Det är en körbar prototyp; verklig modellkvalitet och
hårdvarulatens är ännu inte uppmätta.

## Lokal app

Version 0.4 har ett lokalt webbläsargränssnitt:

```bash
python -m pip install -e '.[app,asr]'
pianissimo-app
```

Installera PyTorch för din hårdvara och starta Ollama med en installerad
instruction-modell. Gränssnittet öppnas på `http://127.0.0.1:7860`. Välj
**Ljudkälla**, ange **Initial kontext**, välj **Språkmodell** under
**Inställningar** och tryck **Starta**. Ljudenheterna kommer från datorn där
appen körs. **Uppdatera** läser om ljudenheter och Ollama-modeller.

**Transkript** visar text efter språkmodellens granskning. Råtext och patchar
finns kvar som interna events; frontend använder `display`. Vid modellfel eller
om texten hinner låsas utan slutförd granskning visas originaltext med ett
uttryckligt meddelande. Ljudet fortsätter fångas medan språkmodellen arbetar.

Kontextvyn visar **Bakgrund**, **Aktuellt ämne** och **Tidigare ämnen**.
**Redigera** öppnar separata utkast till samtliga delar. **Spara** uppdaterar
hela minnet atomiskt; **Avbryt** återgår till den senaste kontexten. Timer- och
modellsvar kan inte skriva över utkasten. Om kontexten ändrats sedan redigeringen
öppnades avvisas sparningen med ett meddelande, så att nytillkomna ämnen inte
försvinner. Manuella ändringar ogiltigförklarar pågående modellsvar. Ursprunglig
initialtext bevaras separat, medan rättad bakgrund får företräde i modellen.

**Stoppa** stänger ljudflödet och transkriberar klart redan köat ljud, inklusive
det sista ofullständiga klippet. Transkriptet ligger kvar. Statusen är
**Stoppar** tills detta är klart; även pågående modellladdning måste avslutas.
En ny **Starta** kan därefter börja en ny inspelning.

Appen binder endast till loopback, har `share=False` och Gradio-analys avstängd.
En aktiv inspelning per app-process. Mikrofonen fångas via PortAudio på samma
dator som Python-processen; ingen ljudinmatning från en annan webbläsardator.
Systemljud kräver en ljudenhet som operativsystemet exponerar som input.
Mikrofon-/modellkvalitet är fortfarande inte hårdvarutestad i utvecklingsmiljön.
UI-komponenter, händelsehanterare och manuell kontextändring testas med dubblar.

För befintliga installationer utan den nya startkommandolänken fungerar också:

```bash
python -m pianissimo_context.app
```

## Språkmodell i Ollama

Rekommenderad startkandidat, kontrollerad 2026-10-03: **`qwen3.5:4b`**.
Det är en bedömning utifrån modellstorlek och stödda funktioner, inte en uppmätt
vinnare för svensk ASR-rättning. Ollama anger cirka 3,4 GB för 4B-taggen.
Som jämförelse anges 6,6 GB för `qwen3.5:9b` och 6,6–9,5 GB för `gemma4:e4b`.
Filstorlek är inte total RAM/VRAM: kontextcache, runtime och Pianissimo tillkommer.
Välj 4B som första test när samma maskin ska köra båda modellerna. Jämför 9B
eller Gemma E4B om mer minne finns och rättningskvaliteten motiverar fördröjningen.

```bash
ollama pull qwen3.5:4b
ollama run qwen3.5:4b
# Avsluta chatten med /bye efter att modellen laddats.
pianissimo-app
```

Appen föredrar den rekommenderade modellen om den är installerad. Du kan välja
annan installerad modell. Inga Ollama-modeller laddas ned automatiskt.
Anropen använder `think: false`, JSON-schema, temperatur 0, 16 384 tokens
kontext, högst 1 024 outputtokens och `keep_alive: "10m"`. Inställningarna är
valda för korta strukturerade svar; de är inte generella benchmarkinställningar.
Thinking avaktiveras via API, inte med `/no_think` i texten.
Modellen anger exakt källtext för varje patch. Brokern beräknar positionen och
avvisar tvetydiga förekomster, så att små modeller slipper räkna teckenindex.

Ett litet syntetiskt texttest finns för jämförelse på din maskin:

```bash
python examples/evaluate_model.py --models qwen3.5:4b qwen3.5:9b gemma4:e4b
```

Installera och värm respektive modell först. Testet använder riktiga lokala
Ollama-anrop och produktens patchregler. Det redovisar exakta rättningar,
felstatus samt median/p95 svarstid. Det är tio textfall, ingen WER-mätning eller
ersättning för en inspelning med Pianissimo igång samtidigt. En timeout räknas
som misslyckat fall, även om originaltexten råkade vara korrekt.

Källor: [Qwen3.5 i Ollama](https://ollama.com/library/qwen3.5),
[Gemma 4 i Ollama](https://ollama.com/library/gemma4),
[Qwen3.5-4B modellkort](https://huggingface.co/Qwen/Qwen3.5-4B),
[Ollama thinking](https://docs.ollama.com/capabilities/thinking),
[Ollama strukturerade svar](https://docs.ollama.com/capabilities/structured-outputs).

## Så följer brokern samtalet

Tre delar av kontexten hålls separat:

1. **Initial bakgrund** – användarens beskrivning finns kvar hela samtalet.
2. **Aktuell kontext** – språkmodellens preliminära ämnesbild omprövas var
   20–30:e sekund när det finns nytt tal. Standard är 25 sekunder, med upp
   till 30 sekunders råtext som underlag.
3. **Samtalsminne** – tidigare ämnen, korta sammanfattningar och belägg
   bevaras under sessionen, även efter åtta ämnesbyten. Till varje modellanrop
   väljs högst åtta ämnen: fyra senaste samt fyra utifrån ordöverlappning med
   aktuell råtext och ämne. Hela arkivet är synligt och redigerbart i appen.
   Urvalet är en enkel heuristik och kan missa en indirekt återkoppling.

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
git clone -b main https://github.com/Jtensetti/pianissimo-context-broker.git
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
ett alias för `live`. stdout ger JSONL-events. Inmatningen väntar inte på LLM; klienten visar endast `display`-events.
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
obegränsad kö som bygger upp fördröjning. I CLI:t avbryter Ctrl+C inspelningen
och låser redan transkriberad text; köade och ofullständiga klipp transkriberas
inte då. Appens **Stoppa** behandlar däremot kvarvarande ljud klart.

Detta är chunkad live-ASR, **inte cache-aware neural streaming**. Klippgränser
kan dela ord; överlappningssammanfogning, diarisation och ordnivå-confidence är
inte implementerade. Råtext kommer efter klippets ASR; språkmodellens rättningar
granskas innan frontend visar texten, utan att blockera nästa mikrofonklipp. Den praktiska fördröjningen
beror på klipplängd, hårdvara och modeller.

## Vad får automatiskt rättas?

* Explicit angivna alias: exempelvis `aj sveden` → `AI Sweden`.
* Kapitalisering, interpunktion och blanksteg med bibehållen bokstavs-/sifferföljd.
* I live-läge: modellen kan klassificera ett lokalt hörfel som `asr_error`,
  med en motivering och confidence minst 0.95. Då krävs inte hög
  stavningslikhet, men rättningen måste även klara kodens oberoende uttalskontroll:
  `aj sveden` → `AI Sweden` är möjligt med stöd av kontext.
  Varje sådan patch får omfatta högst fyra ord och 80 tecken på vardera sidan.
  Sammanhanget hjälper modellen, men är inget akustiskt bevis.
* Även små felskrivningar av **aktuella domäntermer och sammansättningar**,
  inte bara namn. Ersättningen måste finnas i aktiv ordlista; källan måste vara
  6–60 tecken, högst fyra ord, med hög stränglikhet och högst två teckens
  ändringsbudget. Exempel: `transkriberingsmodulen` → `transkriberingsmodellen`
  när den senare är en aktiv term.

Alla lexikala byten, även alias och nära domäntermer, måste vara ljudmässigt
närliggande enligt en försiktig svensk textbaserad uttalsapproximation.
Den bevarar vokaler och ljudföljd, normaliserar vissa stavningsvarianter
(exempelvis w/v, dubbelkonsonanter och AI/aj), och tillåter högst två
ljudnyckeländringar med högst 20 procent relativt avstånd. Mycket korta ord
kräver identisk ljudnyckel. Oförändrade omgivande ord räknas inte in i likheten.
Ändringar kontrolleras även mot ursprunglig råtext, så att flera små rättningar
inte kan glida allt längre från ASR-output.

`ledsen` → `deprimerad` stoppas även med högsta LLM-confidence och klinisk
kontext. Avvisade förslag ändrar inte transkriptet. Detta är en heuristik från
text, **inte fonemigenkänning från ljud eller ett bevis för bibehållen betydelse**.
Närliggande ljud kan betyda olika saker; engelska uttal, dialekter och okända
förkortningar kan också ge för konservativa resultat. Kontexten väljer en
kandidat; modellens motivering kan aldrig ersätta uttalskravet.

Exakt källtext, rätt offset/revision och olåst segment krävs. Patchar som
innehåller siffror får bara ändra blanksteg: bland annat minustecken,
decimaltecken, procenttecken och enheters skiftläge bevaras. Kända svenska
negationer och vissa svenska räkneord skyddas i alla patchtyper.
Överlappande batchar avvisas. LLM-confidence måste
vara minst 0.95, men det är en **heuristik, ingen akustisk sannolikhet**.
Stränglikhet är inte heller ett bevis för vad som sades: nära ord kan ha olika
betydelse. Validera automatisk termrättning på egna inspelningar.

Förslag som inte klarar reglerna för en automatisk rättning blir ett `suggestion`
med `text_changed=false`. De skriver inte om den visade transkriptionen.
`--suggestions-only` stänger av både kontextuella hörfelsrättningar och nära termbyten; uttryckliga alias
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
Ämnesarkivet bevaras i RAM under sessionen. Modellens minnesurval är begränsat
till åtta ämnen och råhistoriken till 30 sekunder, max 200 segment/8 000 tecken
i kontextunderlaget. En ny inspelning börjar med ett nytt arkiv.

Den enklare `Broker` behåller det tidigare API:t och tillåter endast alias och
formateringspatchar som standard. Den har ingen automatisk kontextkadens.

| Event | Klientens åtgärd |
| --- | --- |
| `raw` | Intern råtext för diagnostik; visa inte i frontend |
| `display` | Visa komplett segmenttext, sortera på segment_id; kontrollera review_status |
| `patch` | Intern ändringslogg; frontend väntar på `display` |
| `commit` | Lås och spara segmentets sluttext |
| `context` | Visa background, active_context och remembered_topics |
| `glossary` | Uppdatera ASR-ordlista inför nästa decoding |
| `suggestion` | Oanvänt rättningsförslag för diagnostik; visas inte i appen |
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

Enhetstester täcker patchskydd, granskning före visning, låsning under LLM-anrop, decoder-
återställning, initialtext, kadens, råtextbelägg, ämnesminne, termers livslängd,
ämnesbyte, tystnad och parallell råtextinmatning under kontextanalys.
CI kör kärnan på Python 3.11–3.13. NeMo och språkmodeller testas med dubblar.
**Verklig mikrofonhårdvara, modellinferens, WER-förbättring och live-latens är
inte verifierade i utvecklingsmiljön.**

Modellattribution: [KlangAI/pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv),
Klang AI AB, 2026, CC BY 4.0. Inga modellvikter distribueras av projektet.
Se även [NVIDIA NeMo phrase boosting](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_customization/word_boosting.html).
