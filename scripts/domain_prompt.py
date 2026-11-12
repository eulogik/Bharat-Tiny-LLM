#!/usr/bin/env python3
"""Domain router + per-domain system prompts for Bharat-Tiny-LLM.

The v4 SFT experiments showed that a small, repetitive SFT set can destroy
domain routing: the crop template was applied to snake bites and child
diarrhoea. A 2B model does not need to memorize domain behaviour in its weights
if the system can route the query and give it the right instruction.

This module is keyword-based and has no third-party dependencies. It imports
the sibling protocol matcher so routing and the fixed safety layer cannot
disagree. It is a safety layer, not a classifier benchmark: false positives
are cheap (a slightly different prompt), false negatives on health are not.

Usage
-----
  from domain_prompt import route, system_for
  domain = route("Saanp ne kaata hai")      # -> "health"
  prompt = system_for(text)                  # -> per-domain system prompt
"""

from __future__ import annotations

import re

try:  # Local sibling module; keeps the router and protocol matcher consistent.
    import safety_protocol
except Exception:  # noqa: BLE001
    safety_protocol = None

DOMAINS = ("health", "agriculture", "schemes", "education", "daily")

BASE = (
    "You are Bharat-Tiny-LLM, a helpful offline assistant for India. "
    "Reply in the same language and script the user used: Hinglish, Hindi "
    "(Devanagari), or English. Be concise and practical. Never invent live "
    "prices, weather, dates, scores, or government scheme details; say you "
    "cannot check live data and tell the user where to verify. For health "
    "questions, do not diagnose or prescribe; recommend a doctor, ASHA worker, "
    "or 108 for emergencies. For crop questions, prefer safe cultural/IPM steps "
    "and refer to a local KVK or agriculture officer."
)

# Short hints added to BASE for non-health domains. The full rewritten prompts
# below hurt language matching: the base model answered Hinglish crop questions
# in Devanagari and mixed scripts. Even a one-line crop hint caused 7/40
# agriculture language failures and a math regression, so the shipped system
# uses BASE unchanged for non-health domains. The router's job is only to
# separate health (fixed responses) from everything else.
DOMAIN_HINTS = {
    "agriculture": "",
    "schemes": "",
    "education": "",
    "daily": "",
    "health": "",
}

DOMAIN_PROMPTS = {
    "health": (
        "You are Bharat-Tiny-LLM, a careful health-safety assistant for India. "
        "Reply in the user's language and script. You are NOT a doctor: never "
        "diagnose, never name a medicine or dose, and never tell the user to "
        "take a drug. Give safe first-aid/self-care steps (ORS, rest, clean "
        "water, cooling a burn) and name the danger signs that need urgent care "
        "(breathing difficulty, chest pain, unconsciousness, fits, severe "
        "bleeding, no urine, a lethargic child). Refer to a doctor, ASHA worker "
        "or 108. If the question is about a snake bite, poisoning, stroke or "
        "chest pain, say to call 108 or go to the nearest hospital immediately."
    ),
    "agriculture": (
        "You are Bharat-Tiny-LLM, a careful crop-advisory assistant for India. "
        "Reply in the user's language and script. This is a plant/crop question, "
        "not a human health question: never mention human medicine, bandages or "
        "human symptoms. Ask for a clear photo of the affected leaf (both sides) "
        "and the whole plant, plus crop age, soil, irrigation and recent inputs. "
        "Give 2-3 safe cultural/IPM steps (remove affected parts, improve "
        "drainage/airflow, sticky traps, neem oil per label). Do not invent a "
        "pesticide dose. Refer to a local KVK or agriculture officer for "
        "confirmation."
    ),
    "schemes": (
        "You are Bharat-Tiny-LLM, a government-scheme guide for India. Reply in "
        "the user's language and script. Name the official portal, CSC or office "
        "to use, and list the usual documents/eligibility steps. Never invent an "
        "amount, installment date, deadline or eligibility rule; say the user "
        "must verify on the official site or at a CSC. Warn about agents who "
        "charge a fee."
    ),
    "education": (
        "You are Bharat-Tiny-LLM, a patient study helper for India. Reply in the "
        "user's language and script. Explain step by step in simple language, "
        "use one concrete example, and keep it short. For math, show the "
        "calculation and state the final number. Do not invent exam dates or "
        "official notifications; tell the user to check the official board "
        "website."
    ),
    "daily": (
        "You are Bharat-Tiny-LLM, a practical everyday assistant for India. "
        "Reply in the user's language and script. Be concise and practical: give "
        "a short list of steps, not an essay. Do not invent live prices, "
        "weather, scores, train status or dates; say you cannot check live data "
        "and name the source to verify."
    ),
}

# Animal health is agriculture/veterinary, not human health. Check this before
# HEALTH_RE or "gai ko bukhar" becomes a human-fever answer.
ANIMAL_RE = re.compile(
    r"\b(gai|gaay|gay|bakri\w*|bakra|murgi\w*|murga|murge|poultry|cattle|"
    r"goat|buffalo|bhains|pashu|kutta|kutte|dog|billi|cat|bandar|bandariya|monkey|"
    r"जानवर|गाय|बकरी|मुर्गी|भैंस|पशु|अंडे|कुत्ता|कुत्ते|बिल्ली|बंदर)\b",
    re.I,
)
HEALTH_RE = re.compile(
    r"\b(bukhar|fever|dast|diarrh\w*|ulti|vomit|nausea|saanp|snake|khoon|blood|"
    r"bleed|dard|pain|sar|head|seene|chest|saans|breath|asthma|dengue|malaria|"
    r"typhoid|pneumonia|tb|cancer|diabetes|sugar|bp|pressure|thyroid|"
    r"pregnancy|pregnan\w*|garbhvati|bachcha|bacche|baby|newborn|dawai|medicine|"
    r"dose|doctor|aspatal|hospital|phc|ambulance|108|104|asha|daura|seizure|"
    r"behosh|unconscious|jalan|jalna|jalne|jala|burn|haddi|fracture|chot|injury|"
    r"poison|zeher|platelet|vaccin\w*|teeka|anemia|kamzori|chakkar|dizzy|faint|"
    r"neend|sleep|mental|depress\w*|tanav|ilaaj|treatment|bimari|khansi|zukam|"
    r"ors|dehydrat\w*|wajan|weight|exercise|vyayam|smoking|"
    r"kaan|ear|skin|allergy|khaal|bawaseer|piles|periods|masik|aankh|eye|"
    r"daant|dant|tooth|teeth|kamar|back|ghutna|ghutne|knee|jod|joint|"
    r"sujan|swelling|peshab|urine|mutra|heart|dil|gala|gale|throat|kharash|"
    r"khichkhich|khujli|khajli|itch|"
    r"बुखार|दस्त|उल्टी|सांप|साँप|खून|दर्द|साँस|अस्पताल|डॉक्टर|दवा|बच्चे|जलन|"
    r"जलना|जलने|हड्डी|बीमारी|कमजोरी|चक्कर|खांसी|खाँसी|जुकाम|ओआरएस|गर्भवती|वजन|"
    r"व्यायाम|सिगरेट)\b",
    re.I,
)
SCHEME_RE = re.compile(
    r"\b(yojana|scheme|kisan samman|pm kisan|ayushman|ration|aadhaar|pan card|"
    r"mgnrega|nrega|kcc|kisan credit|mudra|pm awas|awas yojana|ujjwala|jan dhan|"
    r"sukanya|atal pension|e-?shram|scholarship|pension|subsidy|installment|"
    r"kist|beneficiary|eligibility|eligible|apply|aavedan|application|"
    r"documents|kagaz|csc|portal|claim|bima|insurance|status|account|card|"
    r"योजना|आयुष्मान|राशन|आधार|पेंशन|किस्त|आवेदन|दस्तावेज|बीमा|खाता)\b",
    re.I,
)
AGRI_RE = re.compile(
    r"\b(kheti|kisan|fasal|crop|tamatar|tomato|aloo|potato|gehu|gehun|wheat|"
    r"dhan|rice|paddy|bhindi|mirchi|pyaz|sarson|chana|makka|ganna|baingan|"
    r"patte|patton|patta|leaf|leaves|peele|yellow|safed|white|daag|spot|mud|"
    r"curl|sookh|dry|wilt|keede|pest|insect|khad|fertilizer|mitti|soil|paani|"
    r"irrigation|sinchai|mandi|bhav|phool|flower|phal|fruit|fungus|fungal|"
    r"spray|pesticide|kvk|krishi|zameen|bigha|beej|seed|paudha|plant|"
    r"fasal|फसल|खेती|किसान|टमाटर|आलू|गेहूं|गेहूँ|धान|पत्ते|पीले|कीड़े|खाद|"
    r"मिट्टी|सिंचाई|मंडी|भाव|ज़मीन|बीज|पौधा|फूल|फल)\b",
    re.I,
)
EDU_RE = re.compile(
    r"\b(exam|padhai|padh|study|school|college|upsc|ssc|ctet|jee|neet|board|"
    r"class|syllabus|revision|timetable|math|ganit|algebra|geometry|fraction|"
    r"percentage|probability|profit|loss|average|lcm|hcf|english|grammar|"
    r"essay|letter|interview|naukri|career|stream|science|physics|chemistry|"
    r"biology|preparation|prepare|taiyari|pariksha|result|marks|course|degree|"
    r"reasoning|aptitude|current affairs|admit card|photosynthesis|pythagoras|"
    r"theorem|newton|water cycle|respiration|democracy|constitution|gst|"
    r"discount|workers|step by step|samjhao|explain|kitna hota|"
    r"पढ़ाई|परीक्षा|स्कूल|कॉलेज|गणित|अंग्रेजी|तैयारी|नौकरी|समझाओ|सवाल)\b",
    re.I,
)
LIVE_RE = re.compile(
    r"\b(aaj|today|abhi|now|live|taaza|latest|weather|mausam|barish|rain|"
    r"mandi|bhav|rate|price|score|cricket|train|railway|bus|traffic|"
    r"date|kab|when|status|kitna|how much|कब|कितना|मौसम|भाव)\b",
    re.I,
)


EMERGENCY_RE = re.compile(
    r"\b(saanp|snake|seene|chest|saans|breath|behosh|unconscious|daura|seizure|"
    r"khoon|blood|bleed|108|ambulance|poison|zeher|haddi|fracture|"
    r"dam ghut|hoth neele|peshab band|urine nahi|"
    r"सांप|सीने|साँस|बेहोश|खून|जहर)\b",
    re.I,
)

# A bite/scratch from a mammal is a human rabies-risk emergency, not a
# livestock question. Check it before ANIMAL_RE, which also matches "kutta".
BITE_RE = re.compile(
    r"\b(kutta|kutte|kutti|dog|bandar|bandariya|monkey|billi|cat|chooha|chuha|rat|"
    r"fox|lomdi|गीदड़|लोमड़ी)\b.{0,40}"
    r"\b(kaat\w*|kat\w*|bite|khurach\w*|scratch\w*|pakad\w*|noch\w*|नोच)\b",
    re.I,
)
BITE_DEVA_RE = re.compile(
    r"(कुत्ते|कुत्ता|बंदर|बिल्ली|लोमड़ी|गीदड़|चूहा|बंदरिया).{0,25}"
    r"(काट|खरोंच|नोच|डस|पकड़)"
)
ANIMAL_DEVA_RE = re.compile(
    r"(गाय|गायें|बकरी|बकरा|मुर्गी|मुर्गा|भैंस|पशु|जानवर|कुत्ता|कुत्ते|बिल्ली|बंदर|बंदरिया|लोमड़ी|चूहा)"
)


# Devanagari combining marks break \b, so Devanagari gets separate substring
# regexes. Checked before the Latin regexes in route().
HEALTH_DEVA_RE = re.compile(
    r"(बुखार|दस्त|उल्टी|सांप|साँप|खून|दर्द|साँस|अस्पताल|डॉक्टर|दवा|बच्चे|जलन|"
    r"जलना|जलने|हड्डी|बीमारी|कमजोरी|चक्कर|खांसी|खाँसी|जुकाम|ओआरएस|गर्भवती|"
    r"वजन|व्यायाम|सिगरेट|इलाज|दौरा|मिर्गी|जहर|सीने|दिल|कुत्ता|कुत्ते|बंदर|"
    r"बिल्ली|कान|आँख|आंख|दाँत|दांत|कमर|घुटना|पेशाब|सूजन|बवासीर|पीरियड|"
    r"खुजली|चोट|घाव|कट|जला|फोड़ा|दवाई|टॉयलेट|पेट)"
)
AGRI_DEVA_RE = re.compile(
    r"(फसल|खेती|किसान|टमाटर|आलू|गेहूं|गेहूँ|धान|पत्ते|पीले|कीड़े|खाद|मिट्टी|"
    r"सिंचाई|मंडी|भाव|ज़मीन|बीज|पौधा|फूल|फल)"
)
SCHEME_DEVA_RE = re.compile(
    r"(योजना|आयुष्मान|राशन|आधार|पेंशन|किस्त|आवेदन|दस्तावेज|बीमा|खाता|मनरेगा)"
)
EDU_DEVA_RE = re.compile(
    r"(पढ़ाई|परीक्षा|स्कूल|कॉलेज|गणित|अंग्रेजी|तैयारी|नौकरी|समझाओ|निबंध|"
    r"साक्षात्कार|इंग्लिश)"
)


def route(text: str) -> str:
    """Return the domain for a user turn.

    Priority: bite -> animal health -> protocol match -> emergency -> scheme ->
    human health -> crop -> education -> daily.

    A bite from a mammal is checked first because it is a human rabies risk
    even though the sentence also contains an animal word. Livestock questions
    are checked before the protocol matcher so "bakri ke bachche ko dast" gets
    veterinary advice, not human diarrhoea advice. Everything else that matches
    a protocol is health by construction, so the router and the matcher cannot
    disagree.
    """
    if BITE_RE.search(text) or BITE_DEVA_RE.search(text):
        return "health"
    if ANIMAL_RE.search(text) or ANIMAL_DEVA_RE.search(text):
        return "agriculture"
    if safety_protocol is not None and safety_protocol.match(text):
        return "health"
    if EMERGENCY_RE.search(text):
        return "health"
    if re.search(r"(सांप|साँप|सीने|साँस|बेहोश|खून|जहर|दौरा|मिर्गी|दिल)", text):
        return "health"
    if SCHEME_RE.search(text) or SCHEME_DEVA_RE.search(text):
        return "schemes"
    if HEALTH_RE.search(text) or HEALTH_DEVA_RE.search(text):
        return "health"
    if AGRI_RE.search(text) or AGRI_DEVA_RE.search(text):
        return "agriculture"
    if EDU_RE.search(text) or EDU_DEVA_RE.search(text):
        return "education"
    return "daily"


def system_for(text: str) -> tuple[str, str]:
    """Return (domain, system prompt) for a user turn."""
    domain = route(text)
    if domain == "health":
        return domain, DOMAIN_PROMPTS["health"]
    extra = DOMAIN_HINTS.get(domain, "")
    if LIVE_RE.search(text) and domain in ("daily", "agriculture", "schemes"):
        extra += (
            " This question asks for live data. Start by saying you cannot check "
            "live data, then name the official source or office to verify."
        )
    return domain, BASE + extra


if __name__ == "__main__":
    import sys

    for line in sys.stdin:
        line = line.strip()
        if line:
            domain, prompt = system_for(line)
            print(f"{domain}\t{line[:80]}")
