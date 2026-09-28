"""Drop prompts that combine minors with nudity/sexual content, and nudity/naturist/sexualized-body prompts generally."""
import re
MINOR = re.compile(r"\b(teen|teens|teenage|teenager|young girl|little girl|girl child|child|children|kid|kids|minor|underage|schoolgirl|loli|shota|\d{1,2} ?(yo|y/o|year[- ]old))\b", re.I)
SEXUAL = re.compile(r"\b(nude|nudity|naked|naturist|nudist|topless|bare|cleavage|lingerie|bikini|sexy|seductive|erotic|busty|breasts?|boobs?|ti ?ts?|hourglass body|thicc|nsfw|sensual|provocative|undressed|panties|underwear)\b", re.I)
def reason(prompt):
    s = SEXUAL.search(prompt); m = MINOR.search(prompt)
    if s and m: return "minor+sexual"
    if s: return "sexual"
    return None
