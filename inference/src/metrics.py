"""Prometheus metrics for the lang-learn inference service.

All metrics follow the lang_learn_ prefix convention.
Import this module in service.py and call record_* helpers.
"""
from __future__ import annotations

# pyrefly: ignore [missing-import]
from prometheus_client import Counter, Gauge, Histogram, Info

# ---------------------------------------------------------------------------
# Service information
# ---------------------------------------------------------------------------
SERVICE_INFO = Info(
    "lang_learn_service",
    "Information about the language learning service",
)
SERVICE_INFO.info({
    "version": "1.0.0",
    "model": "gemma-4-gguf",
    "framework": "bentoml",
})

# ---------------------------------------------------------------------------
# Request metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "lang_learn_requests_total",
    "Total inference requests",
    ["endpoint", "language", "level", "status"],
)

LATENCY = Histogram(
    "lang_learn_latency_seconds",
    "Inference request latency in seconds",
    ["endpoint", "language", "level"],
    buckets=(0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120),
)

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
CEFR_MATCH_SCORE = Gauge(
    "lang_learn_cefr_match_score",
    "Most recent CEFR match score (0-1)",
    ["endpoint", "language", "level"],
)

CEFR_MATCH_HISTOGRAM = Histogram(
    "lang_learn_cefr_match_score_distribution",
    "Distribution of CEFR match scores",
    ["endpoint", "language", "level"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

CEFR_MISMATCH_COUNT = Counter(
    "lang_learn_cefr_mismatches_total",
    "Total responses that did not match requested CEFR level",
    ["endpoint", "language", "level"],
)

# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------
TOKENS_GENERATED = Counter(
    "lang_learn_tokens_generated_total",
    "Total tokens generated",
    ["endpoint", "language", "level"],
)

# ---------------------------------------------------------------------------
# Resource metrics
# ---------------------------------------------------------------------------
GPU_MEMORY_USED = Gauge(
    "lang_learn_gpu_memory_bytes",
    "GPU memory used by inference container",
)

MODEL_LOADED = Gauge(
    "lang_learn_model_loaded",
    "1 if model loaded and ready, 0 otherwise",
)

# ---------------------------------------------------------------------------
# User feedback
# ---------------------------------------------------------------------------
USER_FEEDBACK = Counter(
    "lang_learn_user_feedback_total",
    "User thumbs up/down on responses",
    ["endpoint", "language", "level", "rating"],
)

# ---------------------------------------------------------------------------
# Helper: CEFR scoring
# ---------------------------------------------------------------------------
_A1_WORDS = frozenset([
    "abend", "abendessen", "aber", "acht", "alle", "allein", "alles", "alt",
    "an", "antworten", "apfel", "apotheke", "arbeit", "arbeiten", "arzt", "auch",
    "auf", "auf Wiedersehen", "auge", "aus", "ausbildung", "auto",
    "bahn", "bahnhof", "ball", "banane", "bank", "baum", "beide", "beim",
    "bein", "beispiel", "beißen", "bekommen", "bereit", "berg", "beruf", "bett",
    "bezahlen", "bier", "bild", "billig", "bin", "birne", "bist", "bitte",
    "blau", "bleiben", "bleistift", "blume", "boden", "brot", "bruder", "buch",
    "bus", "butter", "café", "computer", "da", "danke", "dann", "darf",
    "das", "dein", "dem", "den", "denken", "der", "des", "deutsch",
    "dialog", "dich", "die", "dienstag", "ding", "dir", "doch", "donnerstag",
    "dort", "drei", "du", "dunkel", "durch", "duschen", "durst", "ecke",
    "ei", "ein", "eine", "einem", "einen", "einer", "eines", "einfach",
    "einladen", "einkaufen", "einmal", "eis", "eltern", "ende", "englisch", "entschuldigung",
    "er", "erde", "erst", "es", "essen", "euch", "euer", "fahren",
    "fahrrad", "familie", "fast", "vater", "fehlen", "fenster", "ferien", "fernsehen",
    "fertig", "feuer", "finden", "fisch", "flasche", "fleisch", "fliegen", "flughafen",
    "flugzeug", "fluss", "fragen", "frau", "frei", "freitag", "freizeit", "freund",
    "freundin", "frisch", "froh", "früh", "frühling", "frühstück", "für", "fuß",
    "fußball", "geben", "gegen", "gehen", "gelb", "geld", "gemüse", "genug",
    "gern", "gerne", "gespräch", "gesund", "glauben", "groß", "großeltern", "grün",
    "gut", "guten", "haar", "haben", "habt", "hallo", "hand", "handy",
    "hast", "hat", "hatte", "haus", "heiß", "helfen", "herbst", "herr",
    "heute", "hier", "himmel", "hinter", "hobby", "hoch", "holen", "hören",
    "hotel", "hund", "hunger", "ich", "ihm", "ihn", "ihr", "immer",
    "in", "ist", "ja", "jahr", "jeder", "jetzt", "jung", "kaffee",
    "kalt", "kann", "karte", "kartoffel", "käse", "kaufen", "kein", "keine",
    "kind", "kino", "klasse", "klein", "kochen", "koffer", "kollege", "kommen",
    "können", "kopf", "kosten", "krank", "krankenhaus", "küche", "kuchen", "kugelschreiber",
    "kurz", "lachen", "laden", "land", "lang", "langsam", "laufen", "laut",
    "leben", "legen", "lehrer", "leicht", "leise", "lernen", "lesen", "letzte",
    "leute", "licht", "lieb", "lieben", "lieber", "liegen", "links", "luft",
    "lustig", "machen", "mädchen", "mann", "markt", "mehr", "mein", "mensch",
    "messer", "milch", "minute", "mit", "mitbringen", "mittag", "mittagessen", "mitte",
    "mittwoch", "möchte", "mitnehmen", "monat", "montag", "morgen", "müde", "mund",
    "musik", "müssen", "mutter", "nach", "nachmittag", "nah", "name", "natur",
    "neben", "nehmen", "nein", "nennen", "nett", "neu", "neun", "nicht",
    "nichts", "nie", "noch", "nudeln", "nummer", "nur", "ob", "obst",
    "oder", "offen", "öffnen", "oft", "ohne", "onkel", "orange", "ort",
    "papier", "park", "partner", "person", "pfeffer", "pferd", "platz", "post",
    "preis", "problem", "rad", "regen", "regnen", "reich", "reise", "reisen",
    "restaurant", "richtig", "riechen", "rot", "rufen", "ruhig", "rund", "sache",
    "saft", "sagen", "salat", "salz", "samstag", "sauber", "sauer", "schade",
    "schlafen", "schlecht", "schließen", "schlüssel", "schmecken", "schmutzig", "schnee", "schnell",
    "schon", "schön", "schreiben", "schuh", "schule", "schüler", "schwarz", "schwer",
    "schwester", "schwimmen", "sechs", "see", "sehen", "sehr", "seid", "sein",
    "seit", "seite", "selbst", "senden", "sessel", "sich", "sie", "sieben",
    "sind", "singen", "sitzen", "sohn", "sommer", "sonne", "sonntag", "spaß",
    "spät", "spazieren", "speisekarte", "spielen", "sport", "sprechen", "stadt", "stand",
    "stark", "stehen", "stelle", "stellen", "stift", "straße", "stück", "stuhl",
    "suchen", "suppe", "süß", "tag", "tante", "tasche", "tasse", "tee",
    "telefon", "teller", "teuer", "tief", "tier", "tisch", "tochter", "toll",
    "tomate", "tragen", "treffen", "trinken", "tschüss", "tür", "uhr", "über",
    "um", "und", "uns", "unser", "unter", "urlaub", "verkaufen", "verstehen",
    "viel", "vielleicht", "vier", "vom", "von", "vor", "vormittag", "wagen",
    "wald", "wann", "warm", "warum", "was", "waschen", "wasser", "weg",
    "weich", "wein", "weit", "weiter", "welt", "wenig", "wer", "werden",
    "werfen", "wetter", "wichtig", "wie", "wieder", "will", "wind", "winter",
    "wir", "wissen", "wo", "woche", "wochenende", "woher", "wohin", "wohnen",
    "wohnung", "wollen", "wort", "wunderbar", "zahlen", "zahn", "zehn", "zeigen",
    "zeit", "zeitung", "zimmer", "zu", "zucker", "zug", "zusammen", "zwei",
    "zwischen"
])

_B2_WORDS = frozenset([
    "ablehnen", "absehbar", "absolvieren", "abstimmen", "abwechslungsreich", "allerdings",
    "allmählich", "alltäglich", "analysieren", "anforderung", "angemessen", "ankommen",
    "anerkennen", "anliegen", "annehmen", "anspruch", "anteil", "anwendung",
    "argument", "argumentation", "art", "assoziation", "atmosphäre", "auffassung",
    "aufgabe", "aufheben", "aufklären", "aufmerksam", "aufnahme", "aufwand",
    "ausbildung", "ausdrücken", "auseinandersetzen", "ausgehen", "ausgleich", "ausnahme",
    "ausreichen", "auswirkung", "beabsichtigen", "bedauern", "bedauerlicherweise", "bedeutung",
    "beeinflussen", "beeinträchtigen", "befassen", "befürworten", "begeistern", "begründen",
    "behaupten", "beinahe", "beitrag", "bekanntlich", "belegen", "bemühen",
    "benötigen", "berücksichtigen", "berufstätig", "beschäftigen", "bescheiden", "beschließen",
    "beschränken", "beschweren", "besitzen", "besorgt", "besprechen", "bestätigen",
    "bestehen", "bestimmen", "beteiligen", "betonen", "betrachten", "betreffen",
    "betreuen", "beurteilen", "bevölkerung", "bevorzugen", "bewerten", "bewusst",
    "bewusstsein", "beziehen", "beziehung", "beziehungsweise", "bezug", "bieten",
    "bildung", "bislang", "bitten", "bleiben", "darlegen", "darstellen",
    "dazu", "definieren", "dementsprechend", "demnach", "demzufolge", "dennoch",
    "deutlich", "differenzieren", "diskutieren", "durchführen", "durchsetzen", "ebenso",
    "ehemalig", "eindeutig", "einfluss", "einführen", "eingreifen", "einheit",
    "einschätzen", "einverstanden", "einzeln", "einzigartig", "empfehlen", "engagieren",
    "entdecken", "entgegen", "enthalten", "entscheiden", "entscheidung", "entschließen",
    "entsprechen", "entsprechend", "entstehen", "entwickeln", "entwicklung", "entwerfen",
    "erachten", "erfahren", "erfahrung", "erfassen", "erfinden", "erfolgreich",
    "erforderlich", "erfreulicherweise", "ergänzen", "ergebnis", "erheblich", "erhöhen",
    "erkennen", "erkenntnis", "erklären", "erlauben", "erläutern", "erleichtern",
    "ermöglichen", "ernähren", "ernsthaft", "eröffnen", "erörtern", "erreichen",
    "erscheinen", "erschweren", "ersetzen", "erstklassig", "erwarten", "erwartung",
    "erwerben", "erzielen", "eventuell", "fähig", "fähigkeit", "fazit",
    "feststellen", "finanzieren", "folglich", "fördern", "forderung", "fortbildung",
    "fortschritt", "garantieren", "geeignet", "gefährden", "gegenüber", "gelegenheit",
    "geltend", "gemeinsam", "gemeinschaft", "genießen", "geradezu", "gerecht",
    "gering", "gesamt", "gesellschaft", "gesetzlich", "gestalten", "gewährleisten",
    "gewiss", "gewissermaßen", "gleichwohl", "grenze", "grundlage", "grundsätzlich",
    "häufig", "hauptsächlich", "herausforderung", "hervorragend", "hinsichtlich", "hinweisen",
    "infolgedessen", "infrage", "inhalt", "initiative", "insofern", "insbesondere",
    "interesse", "interpretieren", "investieren", "jedenfalls", "jedoch", "jeweils",
    "keineswegs", "kenntnis", "klarstellen", "komplex", "kompliziert", "konflikt",
    "konsequenz", "konsequent", "kritisieren", "kulturell", "künftig", "lage",
    "langfristig", "lebenslauf", "lediglich", "leisten", "leistung", "letztendlich",
    "maßnahme", "meinung", "merkwürdig", "methode", "mittlerweile", "möglich",
    "möglichkeit", "nachweisen", "nachhaltig", "nämlich", "notwendig", "nichtsdestotrotz",
    "nichtsdestoweniger", "nutzen", "obwohl", "offenbar", "optimieren", "organisieren",
    "perspektive", "planen", "prinzip", "prinzipiell", "prozess", "rahmenbedingung",
    "realisieren", "reagieren", "regelung", "relevant", "risiko", "rolle",
    "schwierig", "schwierigkeit", "selbstverständlich", "sicherlich", "sinnvoll", "somit",
    "sozial", "spezifisch", "ständig", "stattfinden", "stattdessen", "stellungnahme",
    "struktur", "tatsächlich", "teilnehmen", "tendenz", "überprüfen", "übrigens",
    "umsetzen", "umstand", "unbedingt", "uneinig", "unerwartet", "unglücklicherweise",
    "unmittelbar", "unterscheiden", "unterschied", "untersuchen", "unterstützen",
    "verändern", "veränderung", "verantwortung", "verbessern", "verbindung", "verbreiten",
    "vereinbaren", "vereinbarung", "verfügen", "vergleichen", "verhandlung", "verhältnis",
    "verhältnismäßig", "verlangen", "verlauf", "vermeiden", "vermuten", "vermutlich",
    "verringern", "verursachen", "voraussetzung", "vorbereiten", "vorgehen", "vorschlagen",
    "vorteil", "wahrscheinlich", "wechseln", "wesentlich", "widerlegen", "wirkung",
    "wirtschaft", "wissenschaft", "zahlreich", "zielsetzung", "zudem", "zufolge",
    "zufrieden", "zusammenhang", "zuständig", "zustimmen", "zuverlässig", "zweifel",
    "zweifellos", "anforderung", "erwartung"
])

import re as _re


def compute_cefr_score(text: str, target_level: str = "A2") -> float:
    """Estimate a CEFR match score (0.0-1.0) for the generated text.

    Higher = more appropriate for the target level.
    A score of 1.0 means the vocabulary perfectly matches the target.
    """
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0

    total = len(words)
    a1_count = sum(1 for w in words if w in _A1_WORDS)
    b2_count = sum(1 for w in words if w in _B2_WORDS)
    german_chars = len(_re.findall(r"[äöüÄÖÜß]", text))
    dialogue_turns = len(_re.findall(r"Person [AB]:", text))

    # A1/A2: expect high A1 ratio, few B2, some German chars, multiple turns
    a1_ratio = a1_count / total
    b2_ratio = b2_count / total
    has_german = min(german_chars / max(total / 10, 1), 1.0)
    has_turns = min(dialogue_turns / 10.0, 1.0)  # at least 10 turns ideal

    if target_level in ("A1", "A2"):
        score = (
            0.40 * a1_ratio
            + 0.20 * (1.0 - b2_ratio)
            + 0.20 * has_german
            + 0.20 * has_turns
        )
    elif target_level in ("B1", "B2"):
        score = (
            0.20 * a1_ratio
            + 0.30 * b2_ratio
            + 0.25 * has_german
            + 0.25 * has_turns
        )
    else:
        score = 0.5 * has_german + 0.5 * has_turns

    return round(min(max(score, 0.0), 1.0), 3)


def record_inference(
    endpoint: str,
    language: str,
    level: str,
    status: str,
    latency_s: float,
    response_text: str,
    token_count: int = 0,
) -> float:
    """Record all metrics for a completed inference request.

    Returns the CEFR match score.
    """
    REQUEST_COUNT.labels(
        endpoint=endpoint, language=language, level=level, status=status
    ).inc()

    LATENCY.labels(endpoint=endpoint, language=language, level=level).observe(
        latency_s
    )

    if token_count > 0:
        TOKENS_GENERATED.labels(
            endpoint=endpoint, language=language, level=level
        ).inc(token_count)

    cefr_score = compute_cefr_score(response_text, level)

    CEFR_MATCH_SCORE.labels(
        endpoint=endpoint, language=language, level=level
    ).set(cefr_score)

    CEFR_MATCH_HISTOGRAM.labels(
        endpoint=endpoint, language=language, level=level
    ).observe(cefr_score)

    if cefr_score < 0.60:
        CEFR_MISMATCH_COUNT.labels(
            endpoint=endpoint, language=language, level=level
        ).inc()

    return cefr_score
