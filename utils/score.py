from rapidfuzz.distance import Levenshtein


def normalize(txt: str) -> str:
	if not txt:
		return ""
	t = txt.strip().lower()
	# basic cleanup
	repl = {
		"’": "'",
		"،": ",",
		"؛": ";",
		"؟": "?",
	}
	for k, v in repl.items():
		t = t.replace(k, v)
	return t


def best_edit_distance(cand: str, targets: list[str]) -> tuple[int, str]:
	if not targets:
		return (999, "")
	cand_n = normalize(cand)
	best = (999, "")
	for tgt in targets:
		d = Levenshtein.distance(cand_n, normalize(tgt))
		if d < best[0]:
			best = (d, tgt)
	return best


def guess_intent(cand: str) -> str:
	c = normalize(cand)
	if any(k in c for k in ["salam", "slm", "salam 3lik", "as-salam"]):
		return "greet"
	if any(k in c for k in ["labas", "bikher", "bikhir", "hamdullah"]):
		return "how_are_you"
	return "other"


def score_turn(transcript: str, turn: dict) -> dict:
	# distance to any allowed variant
	dist, match = best_edit_distance(transcript, turn.get("targets", []))
	# simple intent
	intent = guess_intent(transcript)
	intent_ok = intent in turn.get("intents", [])
	# choose a tip
	tip = turn.get("tips", ["Mzyan. Keep going."])[0]
	if intent_ok and dist <= 3:
		tip = turn.get("tips", ["Mzyan."])[0]
	elif intent_ok:
		tip = "Good intent. Focus on clearer pronunciation or wording."
	else:
		tip = "Try a greeting like Salam 3likom or a short answer like Labas."
	return {
		"intent": intent,
		"intent_ok": intent_ok,
		"edit_distance": dist,
		"best_match": match,
		"feedback": tip,
	}