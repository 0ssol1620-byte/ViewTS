from pathlib import Path
from typing import List              # ← 추가 (3.7+ 호환)

class CsvLogger:
    _root = Path("runs")             # SequenceRunner 쪽에서 변경 가능

    @classmethod
    def write_rows(cls, filename: str, lines: List[str]) -> None:  # ← 수정
        if not lines:
            return
        path = cls._root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")     # 항상 LF로 끝남
