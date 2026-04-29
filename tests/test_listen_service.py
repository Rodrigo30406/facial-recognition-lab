from eleccia_listen import parse_command_text
from eleccia_listen.service import _best_openwakeword_prediction


def test_parse_lights_on() -> None:
    intent = parse_command_text("Eleccia, enciende la luz")
    assert intent.name == "lights_on"


def test_parse_lights_on_plural() -> None:
    intent = parse_command_text("Eleccia, enciende las luces")
    assert intent.name == "lights_on"


def test_parse_lights_off() -> None:
    intent = parse_command_text("hola eleccia apaga la luz")
    assert intent.name == "lights_off"


def test_parse_lights_off_plural() -> None:
    intent = parse_command_text("Eleccia apaga las luces")
    assert intent.name == "lights_off"


def test_parse_status() -> None:
    intent = parse_command_text("eleccia, estado")
    assert intent.name == "status"


def test_parse_camera_on_with_enciende() -> None:
    intent = parse_command_text("eleccia, enciende la camara")
    assert intent.name == "camera_on"


def test_parse_greeting() -> None:
    intent = parse_command_text("eleccia hola")
    assert intent.name == "greeting"


def test_parse_greeting_hola_then_wake_word() -> None:
    intent = parse_command_text("hola eleccia")
    assert intent.name == "greeting"


def test_parse_wake_only() -> None:
    intent = parse_command_text("eleccia")
    assert intent.name == "wake"


def test_parse_unknown() -> None:
    intent = parse_command_text("quiero cafe")
    assert intent.name == "unknown"


def test_best_openwakeword_prediction() -> None:
    model, score = _best_openwakeword_prediction({"eleccia": 0.12, "jarvis": 0.77})
    assert model == "jarvis"
    assert score == 0.77


def test_parse_with_fuzzy_wake_word_alias() -> None:
    intent = parse_command_text(
        "Elexia enciende la luz",
        wake_word="eleccia",
        wake_word_aliases=("elexia",),
        wake_word_fuzzy_threshold=0.8,
        require_wake_word=True,
    )
    assert intent.name == "lights_on"


def test_parse_requires_wake_word() -> None:
    intent = parse_command_text(
        "enciende la luz",
        wake_word="eleccia",
        wake_word_aliases=("elexia",),
        wake_word_fuzzy_threshold=0.8,
        require_wake_word=True,
    )
    assert intent.name == "no_wakeword"
