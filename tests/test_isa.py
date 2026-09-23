"""opendmp.isa: the tables, the encoder/decoder, the assembler, the facts."""
import itertools
import pathlib

import pytest

from opendmp.cli import main as cli
from opendmp.emu import DmpFault, Machine, Unmodelled
from opendmp.isa import AsmError, Insn, assemble, cost_key, decode, disassemble, encode, to_text
from opendmp.isa import facts
from conftest import RE_REPO

HERE = pathlib.Path(__file__).resolve().parent
CHIPS = ("mpu6050", "icm20948")
TAILS = (b"\x00\x00", b"\x3f\x00", b"\x40\x00", b"\x5f\x00", b"\x81\x00", b"\xc1\x08", b"\xc3\xff")


def _modes(chip):
    return (0, 1, 2, 3, 5, 6, 7) + ((4,) if chip == "icm20948" else ())


@pytest.mark.parametrize("chip", CHIPS)
def test_every_byte_decodes_and_reencodes(chip):
    """decode is total, and encode inverts it, for every first byte, mode and operand tail."""
    for mode, b, tail in itertools.product(_modes(chip), range(256), TAILS):
        buf = bytes([b]) + tail
        insn, n = decode(buf, 0, chip, mode)
        assert 1 <= n <= 3
        assert encode(insn, chip, pc=0) == buf[:n], (mode, hex(b), tail.hex(), insn)


def test_vendor_images_round_trip(corpus):
    """disassemble -> text -> assemble reproduces every vendor image byte for byte."""
    for p in sorted(corpus.glob("*.bin")):
        data = p.read_bytes()
        chip, org = ("icm20948", 0x90) if len(data) > 9000 else ("mpu6050", 0)
        prog = assemble(to_text(disassemble(data, chip, org=org), chip))
        size = 0x4000 if chip == "icm20948" else 0xC00
        assert bytes(prog.image(size)[org:org + len(data)]) == data, p.name


@pytest.mark.parametrize("chip", CHIPS)
def test_cost_key_matches_the_emulator(chip):
    """The listing's cycle class is the one the emulator charges, for everything it executes."""
    checked = 0
    for mode, b, tail in itertools.product(_modes(chip), range(256), TAILS[::2]):
        m = Machine(chip)
        m.load(bytes([b]) + tail + b"\xff" * 4, m.entry)
        m.begin_pass(None)
        m.mode = mode
        m.fifo += b"\x00\x00\x00\x01"                   # something for INC under FIFO routing
        insn, _ = decode(m.mem, m.entry, chip, mode)
        try:
            key = m.step()
        except (Unmodelled, DmpFault):
            continue
        assert key.replace(".fifo", "") == cost_key(insn, mode), (mode, hex(b), insn)
        checked += 1
    assert checked > 2000


def test_tilt_asm_equals_dmpasm_tilt_s():
    ex = HERE.parent / "examples/emu"
    prog = assemble((ex / "tilt.asm").read_text())
    assert len(prog.code()) == 47
    assert prog.symbols == {"const": 0x100, "work": 0x110, "param": 0x120, "join": 0x42D}
    assert prog.image()[0x120:0x124] == bytes.fromhex("10000000")
    if not (RE_REPO / "tools/dmpasm.py").exists():
        pytest.skip("InvensenseDMP not present")
    from opendmp.emu.files import read_program
    assert read_program(ex / "tilt.s", "mpu6050") == prog.code()


def asm(body, chip="mpu6050"):
    return assemble(f".chip {chip}\n{body}\n")


@pytest.mark.parametrize("body,msg", [
    ("add 0,1", "mode register is unknown"),                      # nothing set it at the entry
    ("fmode 1\nmin 0,1", "min needs mode f5-f7"),
    ("fmode 5\nadd 0,1", "add needs mode f1-f3"),
    ("fmode 1\nmac 0,1", "mac needs mode f0"),
    ("fmode 1\nbranch far\n.org 0x500\nfar: end", "outside -32..31"),
    ("fmode 1\nbranch far\n.org 0x423\nfar: end", "outside -32..31"),
    ("fmode 1\nbranch nowhere", "undefined label"),
    ("fmode 4", "fmode takes"),
    ("fmode 1\nmov 4", "k must be 0..3"),
    ("p0page 16", "page must be 0..15"),
    ("fmode 1\njump 0x1000", "not an instruction on the mpu6050"),
])
def test_assembler_refuses_mpu(body, msg):
    with pytest.raises(AsmError, match=msg):
        asm(body)


@pytest.mark.parametrize("body,msg", [
    ("fmode 1\njump 0x1010\nadd 0,1", "mode 4"),                  # after a jump: mode 4
    ("fmode 1\nmode4\nadd 0,1", "mode 4"),
    ("mode4\njump.m4 0x1234", "followed by a byte >= 0x80"),
    ("fmode 1\njump.m4 0x1234", "mode-4 target form"),
    ("fmode 1\njump t\nt: add 0,1", "mode 4"),                    # both ways in arrive in mode 4
    ("fmode 1\nif f\njump t\nfmode 1\nt: add 0,1", "unknown"),    # 4 by the jump, 1 falling through
    ("fmode 1\nloop: dec\nif nf\njump loop\nalways\nadd 0,1", "mode 4"),
    ("fmode 1\ncall s\nadd 0,1\ns: fmode 1\nret", "unknown"),     # the callee may change it
    ("fmode 1\nbranch 0x1010", "not an instruction on the icm20948"),
    ("fmode 1\nshl 0", "shl: s must be 1..31"),
    ("fmode 1\ncmp.eq 0,64", "t must be 0..63"),
])
def test_assembler_refuses_icm(body, msg):
    with pytest.raises(AsmError, match=msg):
        asm(body, "icm20948")


def test_assembler_accepts_well_formed_control_flow():
    p = asm("""
        fmode 1
        always
        dec
        if nf
        branch done          ; every f4 leaves mode 4 (probe95-97)
        fmode 1
        add 0,1
done:   always
        fmode 1
        add 1,1
        end""")
    assert p.code()[:6] == bytes.fromhex("f1d8f9daf402")
    with pytest.raises(AsmError, match="mode 4"):
        asm("fmode 1\nbranch next\nnext: add 0,1")          # the MPU's f4 leaves mode 4 too
    q = asm("""
        fmode 1
loop:   always
        fmode 1              ; the target sets the mode (a jump leaves mode 4)
        dec
        if nf
        jump loop
        fmode 1
        call sub
        fmode 1
        add 0,0
        end
sub:    fmode 1
        mul 0,0
        ret""", "icm20948")
    assert q.symbols["loop"] == 0x1001 and q.code()[:1] == b"\xf1"
    lines = {ln.addr: ln for ln in q.lines}
    assert lines[q.symbols["sub"]].insn == Insn("fmode", (1,))


def test_insn_text_round_trip():
    for chip, text in (("mpu6050", "fmode 0\nmac 3,1,inc\nmacn.m 0,0\nfmode 7\nxor 1,2\nmovx 3,inc\n"
                                   "int.f4 65\nnop.f440\nclr.e5\nxchg02\nif f.alt\nend"),
                       ("icm20948", "fmode 1\ncmp.lt 2,17\nshl 5,adv\nsar 32\nload 3\nlo.next\nmovz\n"
                                    "call 0x2000\nfmode 1\nret\nint\nint.fe")):
        p = asm(text, chip)
        again = asm("\n".join(ln.insn.text() for ln in p.lines), chip)
        assert again.code() == p.code()


def test_disassembly_uses_absolute_targets():
    lines = disassemble(bytes.fromhex("f1d9f402d8fed8ff"), "mpu6050", org=0x400)
    assert [ln.insn.text() for ln in lines][2] == "branch 0x0406"


def test_facts_table():
    ids = [f.id for f in facts.FACTS]
    assert len(ids) == len(set(ids))
    for f in facts.FACTS:
        assert f.status in (facts.HW, facts.VENDOR, facts.INFER, facts.UNMEASURED)
        assert set(f.chips) <= set(CHIPS) and f.chips
        assert f.text and f.source
    assert facts.fact("ram.fifo.icm").status == facts.HW


def test_cli(tmp_path, capsys):
    src = HERE.parent / "examples/emu/tilt.asm"
    out = tmp_path / "tilt.bin"
    assert cli(["asm", str(src), "-o", str(out)]) == 0
    img = out.read_bytes()
    assert len(img) == 0x42F and img[0x400:0x402] == b"\xd8\xf1"
    capsys.readouterr()
    assert cli(["dis", str(out), "--start", "0x400", "--asm"]) == 0
    text = capsys.readouterr().out
    assert "branch 0x042d" in text and "mac 3,1" in text
    assert assemble(text).code() == img[0x400:]
    assert cli(["facts", "--chip", "icm20948"]) == 0
    assert "ram.fifo.icm" in capsys.readouterr().out
