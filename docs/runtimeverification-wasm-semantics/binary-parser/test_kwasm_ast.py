import pytest

from kwasm_ast import KBytes

KBYTES_TEST_DATA = (
    (bytes([0x0, 0x41, 0xff]),        'b"\x00\x41\xff"'),
    (bytes([]),                       'b""'),
    (b'WASM',                         'b"WASM"'),
    (b'foo\xAA\x01barbaz',            'b"foo\xAA\x01barbaz"'),
    (b'foo\xAAbar\x01baz',            'b"foo\xAAbar\x01baz"'),
    (b'abcdefghijklmnopqrstuvwxyz',   'b"abcdefghijklmnopqrstuvwxyz"'),
    (0x11223344556677889900aabbccddeeff.to_bytes(length=16, byteorder='big'),
        'b"\x11\x22\x33\x44\x55\x66\x77\x88\x99\x00\xaa\xbb\xcc\xdd\xee\xff"')
)

@pytest.mark.parametrize(('input', 'expected'), KBYTES_TEST_DATA)
def test_kbytes(input, expected) -> None:
    # When
    t = KBytes(input)

    # Then
    assert t.token == expected
    assert t.sort.name == 'Bytes'
