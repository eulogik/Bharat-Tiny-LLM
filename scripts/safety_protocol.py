#!/usr/bin/env python3
"""Fixed safety protocols for high-stakes health questions.

Why this exists
---------------
A 2B model — even with RAG and a strict prompt — generated dangerous first aid:
it told a snake-bite victim to "band the area for 6-8 hours" and offered to
name a medicine for a child's diarrhoea. Those are not acceptable failure modes.

For a small set of critical intents the system must not generate at all. It
returns a short, vetted protocol with 108 and the "do not" list. The model is
still used for low-risk information (schemes, crops, study, daily life), where
a mistake is recoverable.

This is a product safety layer, not a medical device. The text follows common
WHO/Red Cross/National Health Portal first-aid guidance and deliberately avoids
diagnosis, medicine names and doses.

Usage
-----
  from safety_protocol import match, render
  proto = match("saanp ne kaata hai")
  if proto:
      return render(proto, "hinglish")
"""

from __future__ import annotations

import re

PROTOCOLS = {
    "snake_bite": {
        "match": r"(saanp|saap|saamp|sanp|sānp|snake|serpent|nag|सांप|साँप|साप).{0,60}"
                 r"(kaat|kat|das|डस|काट|bite|laga|lag gaya|chadh|zeher|sujan|सूजन|sunn|सुन्न)|"
                 r"(kaat|kat|das|डस|काट|bite).{0,60}(saanp|saap|saamp|sanp|snake|नाग|सांप|साँप)|"
                 r"(do chhed|chhed wale nishan|fang mark)",
        "hinglish": (
            "Turant 108 par call karein ya nazdiki hospital lein. Ye ek emergency hai.\n\n"
            "Karein:\n"
            "1. Vyakti ko shaant aur still rakhein; zyada hilne na dein.\n"
            "2. Ring, chudi, tight kapde hata dein.\n"
            "3. Kata hua ang dil ke level par rakhein.\n"
            "4. Kaatne ka time note karein.\n\n"
            "Bilkul na karein:\n"
            "- Jagh ko kaatna, chusna, jalan (burn) karna ya barf lagana.\n"
            "- Tight bandage ya tourniquet bandhna.\n"
            "- Desi upay, dawa ya sharaab dena.\n\n"
            "Agar saans phool rahi hai, palak gir rahi hai, ya behoshi ho rahi hai to 108 ko turant batayein."
        ),
        "hi": (
            "तुरंत 108 पर कॉल करें या नज़दीकी अस्पताल ले जाएँ। यह एक इमरजेंसी है।\n\n"
            "करें:\n"
            "1. व्यक्ति को शांत और स्थिर रखें; ज़्यादा हिलने न दें।\n"
            "2. अंगूठी, चूड़ी, तंग कपड़े हटा दें।\n"
            "3. काटा हुआ अंग दिल के स्तर पर रखें।\n"
            "4. काटने का समय नोट करें।\n\n"
            "बिल्कुल न करें:\n"
            "- जगह को काटना, चूसना, जलाना या बर्फ लगाना।\n"
            "- कसकर पट्टी या टूर्निकेट बाँधना।\n"
            "- देसी उपाय, दवा या शराब देना।\n\n"
            "अगर साँस फूल रही है, पलक गिर रही है, या बेहोशी हो रही है तो 108 को तुरंत बताएँ।"
        ),
    },
    "animal_bite": {
        "match": r"\b(kutta|kutte|kutti|dog|bandar|bandariya|monkey|billi|cat|chooha|chuha|rat|"
                 r"fox|lomdi|गीदड़|लोमड़ी)\b.{0,50}"
                 r"\b(kaat\w*|kat\w*|bite|khurach\w*|scratch\w*|pakad\w*|noch\w*|नोच)\b|"
                 r"(कुत्ते|कुत्ता|बंदर|बिल्ली|लोमड़ी|गीदड़|चूहा|बंदरिया).{0,30}"
                 r"(काट|खरोंच|नोच|डस|पकड़)",
        "hinglish": (
            "Kisi bhi janwar ka kaatna ya khurachna rabies ka khatra hai. Turant doctor ya hospital jaayein; 108 tab call karein agar khoon zyada beh raha ho ya vyakti behosh ho.\n\n"
            "Karein:\n"
            "1. Ghav ko 15 minute tak behte paani aur soap se dhoyein.\n"
            "2. Agar antiseptic/iodine ho to lagayein.\n"
            "3. Saaf kapde se dhak lein; tight bandage na karein.\n"
            "4. Kaatne wale janwar ko dekha gaya ho to bata dein, par vaccine ke liye intezaar na karein.\n"
            "5. Doctor se anti-rabies vaccine aur tetanus ke baare mein turant poochhein.\n\n"
            "Bilkul na karein:\n"
            "- Mirchi, haldi, tel, mitti ya jhaad-phook na lagayein.\n"
            "- Ghav ko silna ya band karna doctor ki salah ke bina.\n"
            "- Vaccine ko \"21 din baad\" tak talna.\n\n"
            "Rabies ka koi ilaaj nahi hai; vaccine hi bachav hai. Ghav chhota lage to bhi doctor se milna zaroori hai."
        ),
        "hi": (
            "किसी भी जानवर का काटना या खरोंचना रेबीज़ का खतरा है। तुरंत डॉक्टर या अस्पताल जाएँ; 108 तब कॉल करें अगर खून ज़्यादा बह रहा हो या व्यक्ति बेहोश हो।\n\n"
            "करें:\n"
            "1. घाव को 15 मिनट तक बहते पानी और साबुन से धोएँ।\n"
            "2. अगर antiseptic/iodine हो तो लगाएँ।\n"
            "3. साफ़ कपड़े से ढक लें; tight bandage न करें।\n"
            "4. काटने वाले जानवर को देखा गया हो तो बता दें, पर vaccine के लिए इंतज़ार न करें।\n"
            "5. डॉक्टर से anti-rabies vaccine और tetanus के बारे में तुरंत पूछें।\n\n"
            "बिल्कुल न करें:\n"
            "- मिर्ची, हल्दी, तेल, मिट्टी या झाड़-फूँक न लगाएँ।\n"
            "- घाव को सिलना या बंद करना डॉक्टर की सलाह के बिना।\n"
            "- Vaccine को \"21 दिन बाद\" तक टालना।\n\n"
            "रेबीज़ का कोई इलाज नहीं है; vaccine ही बचाव है। घाव छोटा लगे तो भी डॉक्टर से मिलना ज़रूरी है।"
        ),
    },
    "chest_pain": {
        "match": r"(seene|sine|chest|chhati|chhaati|dil|heart|सीने|छाती|दिल|हृदय).{0,50}"
                 r"(dard|pain|bhaaripan|bharipan|bhaari|भारी|दर्द|dabav|dabaav|pressure|attack|दौरा|daura|jalan|जलन|patthar|पत्थर)|"
                 r"(dard|pain|dabav|dabaav|भारी|दर्द).{0,50}(seene|chest|सीने|छाती|dil|दिल)",
        "hinglish": (
            "Turant 108 par call karein ya nazdiki aspatal lein. Seene ka dard ya bhaaripan heart attack ho sakta hai.\n\n"
            "Karein:\n"
            "1. Vyakti ko baithne ya aaram se letne dein.\n"
            "2. Tight kapde dheele karein.\n"
            "3. Unhe akela na chhodein aur 108 ke instructions follow karein.\n\n"
            "Na karein:\n"
            "- Apne aap koi dawa, aspirin ya dose na dein.\n"
            "- Unhe chalne ya seedhi chadhne na dein.\n\n"
            "Agar saans band ho ya behosh ho jayein, 108 ko batayein aur CPR ke liye instructions lein."
        ),
        "hi": (
            "तुरंत 108 पर कॉल करें। सीने का दर्द या भारीपन हार्ट अटैक हो सकता है।\n\n"
            "करें:\n"
            "1. व्यक्ति को बैठने या आराम से लेटने दें।\n"
            "2. तंग कपड़े ढीले करें।\n"
            "3. उन्हें अकेला न छोड़ें और 108 के निर्देश follow करें।\n\n"
            "न करें:\n"
            "- अपने आप कोई दवा, aspirin या dose न दें।\n"
            "- उन्हें चलने या सीढ़ी चढ़ने न दें।\n\n"
            "अगर साँस बंद हो या बेहोश हो जाएँ, 108 को बताएँ और CPR के लिए निर्देश लें।"
        ),
    },
    "fracture": {
        "match": r"\b(haddi|hadi|bone|fracture)\b.{0,50}(toot|tut|crack|darar|दरार|टूट|bahar aa|बाहर आ)|"
                 r"\b(toot|tut|crack)\b.{0,40}(haddi|bone|fracture)|"
                 r"\b(pair|haath|kamar|kohni|ang|ghutna|takhna|kandha|shoulder)\b.{0,50}"
                 r"(mud|tedha|teda|tut|toot|darar)|"
                 r"(gir|girne|gir gaya|fall|takkar|chot).{0,70}"
                 r"(pair|haath|haddi|kamar|ang|kandha|shoulder|टखना|पैर|हाथ|कंधा).{0,70}"
                 r"(tedha|hil nahi|dard|tut|toot|mud|सूजन|सुन्न)|"
                 r"(हड्डी|पैर|हाथ|टखना|कमर|कोहनी).{0,50}(टूट|मुड़|दरार|टेढ़ा|हिल नहीं)|"
                 r"\b(fracture|bone break)\b",
        "hinglish": (
            "Ang ko waise hi rakhein, hilayein nahi.\n\n"
            "Karein:\n"
            "1. Kisi lakdi ya rolled newspaper se halka support dein; tight na karein.\n"
            "2. Sookhi barf 15-20 minute ke liye kapde mein lapet kar lagayein.\n"
            "3. Ghutna/takhna/haddi ko seedha karne ki koshish na karein.\n\n"
            "Turant aspatal/108: khula ghav, khoon, behoshi, ya ang tedha lag raha ho.\n"
            "Kuch khilayein-pilayein nahi; dawa ka dose khud se na dein."
        ),
        "hi": (
            "अंग को वैसे ही रखें, हिलाएँ नहीं।\n\n"
            "करें:\n"
            "1. किसी लकड़ी या rolled newspaper से हल्का support दें; tight न करें।\n"
            "2. सूखी बर्फ 15-20 मिनट के लिए कपड़े में लपेट कर लगाएँ।\n"
            "3. घुटना/टखना/हड्डी को सीधा करने की कोशिश न करें।\n\n"
            "तुरंत अस्पताल/108: खुला घाव, खून, बेहोशी, या अंग टेढ़ा लग रहा हो।\n"
            "कुछ खिलाएँ-पिलाएँ नहीं; दवा का dose खुद से न दें।"
        ),
    },
    "stroke": {
        "match": r"((?<!heat )\bstroke\b|स्ट्रोक|लकवा|lakwa|paralysis|falij|फालिज|fast symptoms|fast test)|"
                 r"\b(muh|face|chehra)\b.{0,40}(tedha|teda|droop|latak|टेढ़ा|लटक)|"
                 r"(चेहरा|मुँह|मुंह).{0,40}(टेढ़ा|लटक)|"
                 r"\b(haath|hath|arm|hand)\b.{0,50}"
                 r"(kamzor|weak|hil nahi|utha nahi|gir raha|grip|pakad nahi|kaam nahi)|"
                 r"(हाथ|बाजू).{0,50}(कमजोर|सुन्न|हिल नहीं)|"
                 r"\b(bolne|bol|boli|labz|zubaan|speech)\b.{0,40}"
                 r"(dikkat|problem|nahi|तकलीफ|नहीं निकल|nahi nikal|nahi aa)|"
                 r"(बोलने|बोल).{0,40}(तकलीफ|नहीं|दिक्कत)|"
                 r"(ek taraf|one side|आधा शरीर|एक तरफ).{0,60}(kaam nahi|weak|kamzor|सुन्न|काम नहीं)|"
                 r"\b(aankh|eye)\b.{0,40}(dikhna band|blind|धुंधला)|"
                 r"(आँख|आंख).{0,40}(दिखना बंद|अंधा|धुंधला)",
        "hinglish": (
            "Ye stroke ke signs ho sakte hain. Turant 108 par call karein ya nazdiki aspatal lein.\n\n"
            "FAST check:\n"
            "- Face: muh ek taraf tedha?\n"
            "- Arm: ek haath kamzor ya gir raha hai?\n"
            "- Speech: bolne mein dikkat?\n"
            "- Time: lakshan shuru hone ka time note karein.\n\n"
            "Karein: vyakti ko aaram se baithayein/letaayein, tight kapde dheele karein, kuch khilayein-pilayein nahi.\n"
            "Na karein: dawa, aspirin, ya koi dose na dein. Unhe akela na chhodein."
        ),
        "hi": (
            "ये स्ट्रोक के संकेत हो सकते हैं। तुरंत 108 पर कॉल करें।\n\n"
            "FAST जाँच:\n"
            "- Face: मुँह एक तरफ टेढ़ा?\n"
            "- Arm: एक हाथ कमजोर या गिर रहा है?\n"
            "- Speech: बोलने में दिक्कत?\n"
            "- Time: लक्षण शुरू होने का समय नोट करें।\n\n"
            "करें: व्यक्ति को आराम से बैठाएँ/लेटाएँ, तंग कपड़े ढीले करें, कुछ खिलाएँ-पिलाएँ नहीं।\n"
            "न करें: दवा, aspirin या कोई dose न दें। उन्हें अकेला न छोड़ें।"
        ),
    },
    "severe_bleeding": {
        "match": r"(khoon|खून|blood|bleed)(?!.{0,25}(kami|janaach|jaanch|test|group|report))"
                 r".{0,70}(ruk|stop|band|beh|bah|chhut|chhoot|nikal|zyada|bahut|tez|तेज|बहुत|नहीं रुक|fountain|बह)|"
                 r"(chot|cut|kat|कट|चोट|घाव|ghav|wound|nase).{0,70}(khoon|खून|blood|bleed)",
        "hinglish": (
            "Turant 108 par call karein ya nazdiki hospital lein.\n\n"
            "Karein:\n"
            "1. Saaf kapde se ghav par 10-15 minute tak seedha dabav dein.\n"
            "2. Ang ko dil ke level se upar rakhein.\n"
            "3. Kapda bheeg jaye to upar se aur dabav dein; hatayein nahi.\n"
            "4. Agar possible ho to clean gloves use karein.\n\n"
            "Na karein:\n"
            "- Ghav mein mitti, haldi, tel ya toothpaste na lagayein.\n"
            "- Chipki hui cheez ko zor se na kheenchein.\n\n"
            "Agar khoon nahi ruk raha, chot gehri hai, ya vyakti behosh ho raha hai to 108 turant."
        ),
        "hi": (
            "तुरंत 108 पर कॉल करें या नज़दीकी अस्पताल ले जाएँ।\n\n"
            "करें:\n"
            "1. साफ कपड़े से घाव पर 10-15 मिनट तक सीधा दबाव दें।\n"
            "2. अंग को दिल के स्तर से ऊपर रखें।\n"
            "3. कपड़ा भीग जाए तो ऊपर से और दबाव दें; हटाएँ नहीं।\n"
            "4. अगर संभव हो तो clean gloves use करें।\n\n"
            "न करें:\n"
            "- घाव में मिट्टी, हल्दी, तेल या toothpaste न लगाएँ।\n"
            "- चिपकी हुई चीज़ को ज़ोर से न खींचें।\n\n"
            "अगर खून नहीं रुक रहा, चोट गहरी है, या व्यक्ति बेहोश हो रहा है तो 108 तुरंत।"
        ),
    },
    "breathing": {
        "match": r"(saans|sans|साँस|सांस|breath|asthma|अस्थमा|dama|दमा).{0,60}"
                 r"(phool|phul|फूल|dikkat|problem|tez|तेज|nahi|नहीं|band|बंद|ghutan|घुटन|tadap|तड़प|attack|अटैक|takleef|तकलीफ)|"
                 r"(dam|dum|दम).{0,30}(ghut|घुट|band|बंद)|"
                 r"(hoth|honth|होंठ|होठ).{0,20}(neele|nile|नीले)",
        "hinglish": (
            "Turant 108 par call karein ya nazdiki hospital lein.\n\n"
            "Karein:\n"
            "1. Vyakti ko seedha baithayein; letne na dein.\n"
            "2. Tight kapde dheele karein aur hawa aane dein.\n"
            "3. Agar unka prescribed inhaler hai to unke instructions ke hisaab se use karne mein madad karein.\n"
            "4. Unhe akela na chhodein.\n\n"
            "Na karein:\n"
            "- Naya inhaler, dawa ya dose khud se na dein.\n"
            "- Unhe khana ya paani na dein agar saans bahut zyada phool rahi hai.\n\n"
            "Agar hoth neele pad rahe hain, bol nahi pa rahe, ya behosh ho rahe hain to 108 turant."
        ),
        "hi": (
            "तुरंत 108 पर कॉल करें या नज़दीकी अस्पताल ले जाएँ।\n\n"
            "करें:\n"
            "1. व्यक्ति को सीधा बैठाएँ; लेटने न दें।\n"
            "2. तंग कपड़े ढीले करें और हवा आने दें।\n"
            "3. अगर उनका prescribed inhaler है तो उनके निर्देशों के हिसाब से use करने में मदद करें।\n"
            "4. उन्हें अकेला न छोड़ें।\n\n"
            "न करें:\n"
            "- नया inhaler, दवा या dose खुद से न दें।\n"
            "- उन्हें खाना या पानी न दें अगर साँस बहुत ज़्यादा फूल रही है।\n\n"
            "अगर होंठ नीले पड़ रहे हैं, बोल नहीं पा रहे, या बेहोश हो रहे हैं तो 108 तुरंत।"
        ),
    },
    "poisoning": {
        "match": r"(zeher|zahar|जहर|poison|विष|phenyl|फिनाइल|keetnashak|कीटनाशक|insecticide|"
                 r"pesticide|kerosene|mitti ka tel|मिट्टी का तेल|shishi|शीशी|spray|स्प्रे|chuha marne|"
                 r"चूहा मारने|sulphas|sulfas|overdose).{0,60}"
                 r"(kha|khaa|pi|pee|खा|पी|nigal|निगल|drink|eat|liya|लिया|chala gaya|चला गया|"
                 r"muh mein|मुँह में|मुंह में|ulti|उल्टी)|"
                 r"(zeher|zahar|जहर|poison|विष)",
        "hinglish": (
            "Turant 108 par call karein ya nazdiki hospital lein.\n\n"
            "Karein:\n"
            "1. Jo cheez li hai uska packet/box saath lein.\n"
            "2. Vyakti ko shaant rakhein.\n"
            "3. Agar behosh hai to kuch bhi muh mein na dein; unhe leta kar karvat par rakhein.\n\n"
            "Na karein:\n"
            "- Apne aap ulti karane ki koshish na karein.\n"
            "- Doodh, paani, tel ya koi dawa na pilayein.\n"
            "- Unhe chalne ya akela chhodne na dein.\n\n"
            "108 ko batayein: kya liya, kitna, kab, aur vyakti ki condition."
        ),
        "hi": (
            "तुरंत 108 पर कॉल करें या नज़दीकी अस्पताल ले जाएँ।\n\n"
            "करें:\n"
            "1. जो चीज़ ली है उसका packet/box साथ लें।\n"
            "2. व्यक्ति को शांत रखें।\n"
            "3. अगर बेहोश है तो कुछ भी मुँह में न दें; उन्हें लेटा कर करवट पर रखें।\n\n"
            "न करें:\n"
            "- अपने आप उल्टी कराने की कोशिश न करें।\n"
            "- दूध, पानी, तेल या कोई दवा न पिलाएँ।\n"
            "- उन्हें चलने या अकेला छोड़ने न दें।\n\n"
            "108 को बताएँ: क्या लिया, कितना, कब, और व्यक्ति की condition।"
        ),
    },
    "burns": {
        "match": r"\b(jal|jala|jalna|jalne|burn|burns|scald)\b.{0,50}"
                 r"(gaya|gaye|gayi|liya|hath|haath|pani|oil|aag|chhala|blister|first aid|upay)|"
                 r"\b(burn|scald)\b.{0,40}(first aid|upay|chhala|blister)|"
                 r"(जल|जला|जलन|जलने|जले).{0,50}(गया|गई|गए|लिया|हाथ|पानी|तेल|आग|छाला|उपाय)|"
                 r"(आग|तेल|पानी).{0,30}(जल|जला|जलन)|"
                 r"\b(chhala|blister)\b.{0,30}\b(jalan|jal|burn)\b",
        "hinglish": (
            "Karein:\n"
            "1. Jali jagah ko 20 minute tak thande behte paani ke neeche rakhein.\n"
            "2. Gehne aur tight kapde hata dein (jo chipak rahe hain unhe na kheenchein).\n"
            "3. Saaf sookhe kapde ya clean cling film se dhak lein.\n\n"
            "Na karein:\n"
            "- Toothpaste, ghee, tel, haldi, barf ya malai na lagayein.\n"
            "- Chale/blister ko na phodein.\n\n"
            "Turant 108/doctor: agar jalan badi hai, chehra/haath/joint par hai, bachcha hai, ya saans mein dhuaan gaya hai."
        ),
        "hi": (
            "करें:\n"
            "1. जली जगह को 20 मिनट तक ठंडे बहते पानी के नीचे रखें।\n"
            "2. गहने और तंग कपड़े हटा दें (जो चिपक रहे हैं उन्हें न खींचें)।\n"
            "3. साफ सूखे कपड़े या clean cling film से ढक लें।\n\n"
            "न करें:\n"
            "- Toothpaste, घी, तेल, हल्दी, बर्फ या मलाई न लगाएँ।\n"
            "- छाले को न फोड़ें।\n\n"
            "तुरंत 108/doctor: अगर जलन बड़ी है, चेहरा/हाथ/joint पर है, बच्चा है, या साँस में धुआँ गया है।"
        ),
    },
    "heat_stroke": {
        "match": r"\b(loo|लू)\b.{0,50}(lag|लग|behosh|बेहोश|chakkar|चक्कर)|"
                 r"\b(garmi|dhoop|heat|गर्मी|धूप)\b.{0,70}"
                 r"(behosh|बेहोश|chakkar|चक्कर|ulti|उल्टी|confus|thanda nahi|badan garam|hosh nahi|होश नहीं|tez|तेज|sust|सुस्त|exhaust)|"
                 r"(गर्मी|धूप|लू).{0,40}(बेहोश|चक्कर|उल्टी|तेज़|तेज|सुस्त|बदहवास)|"
                 r"\b(heat ?stroke|heat exhaustion|loo lag|लू लग)\b",
        "hinglish": (
            "Turant 108 par call karein agar behoshi, confusion, daura, ya bahut tez dhadkan hai.\n\n"
            "Karein:\n"
            "1. Turant chhaya/thandi jagah lein.\n"
            "2. Extra kapde hataayein.\n"
            "3. Geela kapda aur pankhe se shareer thanda karein.\n"
            "4. Agar vyakti hosh mein hai to paani/ORS thoda-thoda piyein.\n\n"
            "Na karein:\n"
            "- Unhe akela na chhodein.\n"
            "- Bahut thanda paani ek saath na pilayein.\n"
            "- Sharaab/chai/coffee na dein."
        ),
        "hi": (
            "तुरंत 108 पर कॉल करें अगर बेहोशी, confusion, दौरा, या बहुत तेज़ धड़कन है।\n\n"
            "करें:\n"
            "1. तुरंत छाया/ठंडी जगह लें।\n"
            "2. Extra कपड़े हटाएँ।\n"
            "3. गीला कपड़ा और पंखे से शरीर ठंडा करें।\n"
            "4. अगर व्यक्ति होश में है तो पानी/ORS थोड़ा-थोड़ा पिएँ।\n\n"
            "न करें:\n"
            "- उन्हें अकेला न छोड़ें।\n"
            "- बहुत ठंडा पानी एक साथ न पिलाएँ।\n"
            "- शराब/चाय/coffee न दें।"
        ),
    },
    "seizure": {
        "match": r"(daura|दौरा|seizure|mirgi|मिर्गी|मिरगी|fits).{0,50}"
                 r"(aaya|आया|aa gaya|aa gaye|ho|हो|pad|पड़|gaya|गया|chal|चल|minute|मिनट|attack|अटैक|hua|हुआ)|"
                 r"(jhatke|झटके).{0,40}(aa rahe|आ रहे|hosh nahi|होश नहीं)|"
                 r"(kaanpne|kaanp|कांपने|कांप|kampan|कंपन).{0,40}"
                 r"(laga|लगा|shuru|शुरू|raha|रहा|hai|है)|"
                 r"(muh se jhaag|मुँह से झाग|मुंह से झाग)|"
                 r"(akad gaya|अकड़ गया).{0,40}(aankh|आँख|आंख)|"
                 r"(daura|दौरा|मिर्गी|मिरगी|seizure|fits)",
        "hinglish": (
            "Karein:\n"
            "1. Aas-paas ki cheezein hataayein; sir ke neeche kuch naram rakhein.\n"
            "2. Time note karein (kitne minute chala).\n"
            "3. Daura khatam hone tak unhe karvat par rakhein.\n\n"
            "Na karein:\n"
            "- Unhe pakad kar rokne ki koshish na karein.\n"
            "- Muh mein chammach, ungli, paani ya dawa na daalein.\n\n"
            "Turant 108: agar daura 5 minute se zyada chale, baar-baar aaye, saans mein dikkat ho, chot lage, ya pehli baar daura aaya ho."
        ),
        "hi": (
            "करें:\n"
            "1. आस-पास की चीज़ें हटाएँ; सिर के नीचे कुछ नरम रखें।\n"
            "2. Time नोट करें (कितने मिनट चला)।\n"
            "3. दौरा खत्म होने तक उन्हें करवट पर रखें।\n\n"
            "न करें:\n"
            "- उन्हें पकड़ कर रोकने की कोशिश न करें।\n"
            "- मुँह में चम्मच, उँगली, पानी या दवा न डालें।\n\n"
            "तुरंत 108: अगर दौरा 5 मिनट से ज़्यादा चले, बार-बार आए, साँस में दिक्कत हो, चोट लगे, या पहली बार दौरा आया हो।"
        ),
    },
    "child_fever": {
        "match": r"(bachcha|bachche|bacche|bacchi|baby|shishu|बच्चा|बच्चे|बच्ची|शिशु|newborn|नवजात|toddler|infant).{0,70}"
                 r"(bukhar|fever|बुखार|garmi|गर्मी|102|103|104|101)|"
                 r"(bukhar|fever|बुखार).{0,50}(bachcha|bacche|baby|बच्चा|बच्चे|शिशु)",
        "hinglish": (
            "Main dawai ya dose nahi bata sakta. Ghar par:\n"
            "1. Halke kapde pehnaayein, kamra thanda rakhein.\n"
            "2. Paani/ORS thoda-thoda pilayein; breastfeeding jaari rakhein.\n"
            "3. Temperature note karein.\n\n"
            "Turant doctor/108: agar baccha 3 mahine se chhota hai, bukhar 3 din se zyada hai, saans phool rahi hai, daura aaya hai, shareer par laal daag hain, ya baccha bahut sust/behosh hai.\n\n"
            "Aspirin bachche ko kabhi na dein. Kisi bhi dawai ka dose doctor se poochhein."
        ),
        "hi": (
            "मैं दवा या dose नहीं बता सकता। घर पर:\n"
            "1. हल्के कपड़े पहनाएँ, कमरा ठंडा रखें।\n"
            "2. पानी/ORS थोड़ा-थोड़ा पिलाएँ; breastfeeding जारी रखें।\n"
            "3. Temperature नोट करें।\n\n"
            "तुरंत doctor/108: अगर बच्चा 3 महीने से छोटा है, बुखार 3 दिन से ज़्यादा है, साँस फूल रही है, दौरा आया है, शरीर पर लाल दाग हैं, या बच्चा बहुत सुस्त/बेहोश है।\n\n"
            "Aspirin बच्चे को कभी न दें। किसी भी दवा का dose doctor से पूछें।"
        ),
    },
    "diarrhea": {
        "match": r"(dast|दस्त|diarrh\w*|loose motion|loose motions|लूज मोशन)|"
                 r"(pet|पेट).{0,30}(kharab|खराब).{0,50}(latrine|potty|टॉयलेट|motion|दस्त)|"
                 r"(\d+\s*baar|baar baar|bar bar|बार-बार).{0,40}(latrine|potty|टॉयलेट|motion)|"
                 r"(latrine|potty|टॉयलेट).{0,30}(khoon|blood|खून)",
        "hinglish": (
            "ORS (Oral Rehydration Solution) sabse zaroori hai:\n"
            "1. Ek ORS packet ko packet par likhe saaf paani mein milaayein.\n"
            "2. Thoda-thoda kar ke baar-baar pilayein.\n"
            "3. Bachche ko breastfeeding jaari rakhein.\n"
            "4. Zinc ke liye doctor/ASHA se poochhein (bachche ke liye).\n\n"
            "Turant doctor/108: khoon aa raha hai, peshab band ho gaya hai, aankhein dhansi hui hain, bahut kamzori/chakkar hai, bukhar tez hai, ya baccha sust hai.\n\n"
            "Bina doctor ke antibiotic ya dast rokne wali dawa na lein. Dose main nahi bata sakta."
        ),
        "hi": (
            "ORS (Oral Rehydration Solution) सबसे ज़रूरी है:\n"
            "1. एक ORS packet को packet पर लिखे साफ़ पानी में मिलाएँ।\n"
            "2. थोड़ा-थोड़ा करके बार-बार पिलाएँ।\n"
            "3. बच्चे को breastfeeding जारी रखें।\n"
            "4. Zinc के लिए doctor/ASHA से पूछें (बच्चे के लिए)।\n\n"
            "तुरंत doctor/108: खून आ रहा है, पेशाब बंद हो गया है, आँखें धँसी हुई हैं, बहुत कमज़ोरी/चक्कर है, बुखार तेज़ है, या बच्चा सुस्त है।\n\n"
            "बिना doctor के antibiotic या दस्त रोकने वाली दवा न लें। Dose मैं नहीं बता सकता।"
        ),
    },
}


def match(text: str) -> str | None:
    """Return the protocol key for a critical health query, or None."""
    for key, item in PROTOCOLS.items():
        if re.search(item["match"], text, re.I):
            return key
    return None


def render(key: str, language: str = "hinglish") -> str:
    item = PROTOCOLS[key]
    if language == "hi":
        return item.get("hi", item["hinglish"])
    return item["hinglish"]


def detect_language(text: str) -> str:
    deva = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "hi" if deva > latin else "hinglish"


GENERIC_HEALTH = {
    "hinglish": (
        "Main doctor nahi hoon aur lakshan dekh kar diagnosis ya dawai nahi bata sakta. "
        "Kripya nazdiki doctor, ASHA worker ya PHC se milein.\n\n"
        "Turant 108/aspatal: saans phool rahi ho, seene mein dard ho, behoshi ho rahi ho, "
        "daura aaya ho, bahut tez bukhar ho, khoon na ruk raha ho, ya baccha bahut sust ho.\n\n"
        "Kisi bhi dawai ka dose doctor se poochhein; bina doctor ke antibiotic ya painkiller na lein."
    ),
    "hi": (
        "मैं डॉक्टर नहीं हूँ और लक्षण देखकर diagnosis या दवा नहीं बता सकता। "
        "कृपया नज़दीकी डॉक्टर, ASHA worker या PHC से मिलें।\n\n"
        "तुरंत 108/अस्पताल: साँस फूल रही हो, सीने में दर्द हो, बेहोशी हो रही हो, "
        "दौरा आया हो, बहुत तेज़ बुखार हो, खून न रुक रहा हो, या बच्चा बहुत सुस्त हो।\n\n"
        "किसी भी दवा का dose डॉक्टर से पूछें; बिना डॉक्टर के antibiotic या painkiller न लें।"
    ),
}

# Crop answers must never mention human medicine, 108 or human clinical care.
# The base model does this when it confuses a plant symptom with a human one
# ("phool khaal" -> "108/ASHA/doctor", "ghee lagayein"). The guard replaces the
# answer with a vetted crop-triage response.
CROP_UNSAFE_RE = re.compile(
    r"\b(paracetamol|ibuprofen|antibiotic|bandage|doctor|hospital|asha|108|"
    r"doodh|ghee|दूध|घी|डॉक्टर|अस्पताल|इंसान|insaan|skin)\b|"
    r"\b(patte|gehu|aloo|paudhe|phool|patti) ko .{0,40}\b(dho|dhoye|dhoyen|dhona|nahl\w*)\b|"
    r"\bgaram paani\b.{0,30}\b(patte|paudhe|phool)\b",
    re.I,
)
CROP_FALLBACK = {
    "hinglish": (
        "Bhai, yeh paudhon ka problem hai, insaan ka nahi — koi human medicine, "
        "bandage ya 108 nahi.\n\n"
        "Pehle ek saaf photo bhejiye — patte ke upar aur neeche dono side, aur "
        "paudhe ka poora hissa, saath mein mitti/paani ki jankari.\n\n"
        "Tab tak: 1) affected hisse hata kar khet se door dispose karein, "
        "2) paani subah dein, 3) hawa aane dein, 4) sticky trap/neem oil label "
        "ke hisaab se try karein. Koi pesticide bina KVK/agri officer ki salah ke "
        "na chhidkein. Nazdiki KVK se sample dikhayein."
    ),
    "hi": (
        "भाई, यह पौधों की समस्या है, इंसान की नहीं — कोई human medicine, bandage "
        "या 108 नहीं।\n\n"
        "पहले एक साफ़ फोटो भेजिए — पत्ते के ऊपर और नीचे दोनों तरफ, और पौधे का "
        "पूरा हिस्सा, साथ में मिट्टी/पानी की जानकारी।\n\n"
        "तब तक: 1) प्रभावित हिस्से हटाकर खेत से दूर Dispose करें, 2) पानी सुबह दें, "
        "3) हवा आने दें, 4) sticky trap/neem oil label के हिसाब से try करें। कोई "
        "pesticide बिना KVK/कृषि अधिकारी की सलाह के न छिड़कें। नज़दीकी KVK से "
        "sample दिखाएँ।"
    ),
}


# Animal health is not crop health. The base model often answers "gai ko
# bukhar" with either human medicine or crop advice; use a veterinary referral.
ANIMAL_RE = re.compile(
    r"\b(gai|gaay|gay|bakri|bakra|murgi|murga|murge|poultry|cattle|goat|"
    r"buffalo|bhains|pashu|जानवर|गाय|बकरी|मुर्गी|भैंस|पशु)\b",
    re.I,
)
ANIMAL_FALLBACK = {
    "hinglish": (
        "Yeh janwar ki samasya hai. Pashu-chikitsak (veterinary doctor) se milein; "
        "insaan ki dawai ya dose na dein.\n\n"
        "Pashu ko saaf paani aur aaram dein. Agar bukhar tez hai, khaana band kar "
        "diya hai, dast/khoon hai, ya pashu bahut sust hai to jaldi nazdiki Pashu "
        "Chikitsalaya ya KVK lein."
    ),
    "hi": (
        "यह जानवर की समस्या है। पशु-चिकित्सक (veterinary doctor) से मिलें; "
        "इंसान की दवा या dose न दें।\n\n"
        "पशु को साफ़ पानी और आराम दें। अगर बुखार तेज़ है, खाना बंद कर दिया है, "
        "दस्त/खून है, या पशु बहुत सुस्त है तो जल्दी नज़दीकी पशु चिकित्सालय या KVK ले जाएँ।"
    ),
}


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+|\n+")


NONHEALTH_EMERGENCY_RE = re.compile(r"\b(108|102|104)\b|ambulance|एम्बुलेंस", re.I)
QUERY_EMERGENCY_RE = re.compile(
    r"\b(saanp|snake|seene|chest|saans|breath|behosh|unconscious|daura|seizure|"
    r"khoon|blood|accident|poison|zeher|haddi|fracture|jalan|burn|chot|injury)\b|"
    r"(सांप|साँप|सीने|साँस|बेहोश|खून|दौरा|जहर|हड्डी|जलन|चोट)",
    re.I,
)
NONHEALTH_FALLBACK = {
    "hinglish": (
        "Main iska pakka jawab nahi de sakta. Kripya official portal, CSC ya "
        "sambandhit office se confirm karein. 108 sirf medical emergency ke liye hai."
    ),
    "hi": (
        "मैं इसका पक्का जवाब नहीं दे सकता। कृपया official portal, CSC या "
        "संबंधित कार्यालय से पुष्टि करें। 108 सिर्फ़ medical emergency के लिए है।"
    ),
}


def guard_nonhealth_emergency(answer: str, query: str, language: str = "hinglish") -> tuple[str, bool]:
    """Remove 108/ambulance advice from a non-health answer.

    A ration-card or scheme answer must never tell the user to call 108. If the
    query itself is an emergency, leave it alone; this guard is only for
    non-health domains.
    """
    if QUERY_EMERGENCY_RE.search(query or ""):
        return answer, False
    if not NONHEALTH_EMERGENCY_RE.search(answer or ""):
        return answer, False
    lang = language if language == "hi" else "hinglish"
    parts = [p for p in SENTENCE_SPLIT_RE.split(answer) if p.strip()]
    kept = [p for p in parts if not NONHEALTH_EMERGENCY_RE.search(p)]
    cleaned = " ".join(kept).strip()
    if len(cleaned.split()) < 20:
        return NONHEALTH_FALLBACK[lang], True
    note = (
        "Note: 108 sirf medical emergency ke liye hai, is kaam ke liye nahi."
        if lang != "hi"
        else "नोट: 108 सिर्फ़ medical emergency के लिए है, इस काम के लिए नहीं।"
    )
    return cleaned + "\n\n" + note, True


def guard_crop(answer: str, language: str = "hinglish", query: str = "") -> tuple[str, bool]:
    """Replace/trim a crop/animal answer that leaked human-medical advice."""
    lang = language if language == "hi" else "hinglish"
    if query and ANIMAL_RE.search(query):
        return ANIMAL_FALLBACK[lang], True
    if CROP_UNSAFE_RE.search(answer):
        parts = [p for p in SENTENCE_SPLIT_RE.split(answer) if p.strip()]
        kept = [p for p in parts if not CROP_UNSAFE_RE.search(p)]
        cleaned = " ".join(kept).strip()
        if len(cleaned.split()) >= 25:
            return cleaned, True
        return CROP_FALLBACK[lang], True
    return answer, False



def health_answer(text: str) -> tuple[str, dict]:
    """Return a fixed, vetted health response. No model generation."""
    key = match(text)
    lang = detect_language(text)
    if key:
        return render(key, lang), {"protocol": key, "language": lang}
    return GENERIC_HEALTH[lang], {"protocol": None, "language": lang}



if __name__ == "__main__":
    import sys

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        key = match(line)
        print(f"{key or '-'}\t{line[:80]}")
        if key:
            print(render(key, detect_language(line)))
