"""Exam Orchestrator: Concurrently coordinates Teil generation, Redis storage, and evaluation.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import uuid
from typing import Any, Dict, List

from .evaluators import evaluate_reading, evaluate_writing
from .generators import (
    generate_lesen_teil1,
    generate_lesen_teil2,
    generate_lesen_teil3,
    generate_lesen_teil4,
    generate_schreiben_teil1,
    generate_schreiben_teil2,
)
from .models import ExamModule, ExamPaper
from .storage import ExamStorage

logger = logging.getLogger("lang_learn.exam.orchestrator")


class ExamOrchestrator:
    """Coordinates generation, persistence, and evaluation of Goethe A2 exams."""

    def __init__(self, storage: ExamStorage | None = None) -> None:
        self.storage = storage or ExamStorage()

    async def generate_paper(self, module: str = "lesen", level: str = "A2") -> Dict[str, Any]:
        """Generate a complete exam paper for the requested module.

        Strictly executes only the generators corresponding to `module`.
        Uses `asyncio.gather` for parallel generation of all Teile.
        """
        mod = module.strip().lower()
        paper_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        if mod == "lesen":
            # Concurrent fan-out for all 4 reading Teile
            results = await asyncio.gather(
                generate_lesen_teil1(level),
                generate_lesen_teil2(level),
                generate_lesen_teil3(level),
                generate_lesen_teil4(level),
                return_exceptions=True
            )

            teils = {}
            aggregated_answer_key = {}
            aggregated_explanations = {}

            for idx, res in enumerate(results, start=1):
                if isinstance(res, Exception):
                    logger.error("Teil %d generation error: %s", idx, res)
                    # Retry single Teil once
                    if idx == 1:
                        t_data, t_key = await generate_lesen_teil1(level)
                    elif idx == 2:
                        t_data, t_key = await generate_lesen_teil2(level)
                    elif idx == 3:
                        t_data, t_key = await generate_lesen_teil3(level)
                    else:
                        t_data, t_key = await generate_lesen_teil4(level)
                else:
                    t_data, t_key = res

                t_name = f"teil{idx}"
                teils[t_name] = t_data
                aggregated_answer_key.update(t_key.get("answer_key", {}))
                aggregated_explanations.update(t_key.get("explanations", {}))

            full_answer_key = {
                "answer_key": aggregated_answer_key,
                "explanations": aggregated_explanations
            }

            paper = ExamPaper(
                paper_id=paper_id,
                module=ExamModule.LESEN,
                level=level,
                created_at=created_at,
                duration_minutes=30,
                total_points=25.0,
                teils=teils,
                answer_key=full_answer_key
            )

        elif mod == "schreiben":
            # Concurrent fan-out for 2 writing Teile
            results = await asyncio.gather(
                generate_schreiben_teil1(level),
                generate_schreiben_teil2(level),
                return_exceptions=True
            )

            teils = {}
            for idx, res in enumerate(results, start=1):
                if isinstance(res, Exception):
                    logger.error("Schreiben Teil %d generation error: %s", idx, res)
                    if idx == 1:
                        t_data = await generate_schreiben_teil1(level)
                    else:
                        t_data = await generate_schreiben_teil2(level)
                else:
                    t_data = res

                t_name = f"teil{idx}"
                teils[t_name] = t_data

            paper = ExamPaper(
                paper_id=paper_id,
                module=ExamModule.SCHREIBEN,
                level=level,
                created_at=created_at,
                duration_minutes=30,
                total_points=25.0,
                teils=teils,
                answer_key=None
            )
        else:
            raise ValueError(f"Unknown exam module: '{module}'. Expected 'lesen' or 'schreiben'.")

        # Persist full paper with answer key in Redis
        label = self.storage.store_paper(paper_id, paper.model_dump())

        # Return sanitized paper to client (no server-side answer keys)
        client_paper = {
            "paper_id": paper.paper_id,
            "label": label,
            "module": paper.module.value,
            "level": paper.level,
            "created_at": paper.created_at,
            "duration_minutes": paper.duration_minutes,
            "total_points": paper.total_points,
            "teils": paper.teils
        }
        return client_paper

    async def generate_paper_stream(self, module: str = "lesen", level: str = "A2"):
        """Progressively generate and yield exam Teile one-by-one as Server-Sent Events.

        Yields:
          - {"type": "init", "paper_id": ..., "module": ..., "total_teils": ...}
          - {"type": "teil", "teil_index": 1, "teil_name": "teil1", "data": {...}}
          ...
          - {"type": "done", "paper": {...}}
        """
        mod = module.strip().lower()
        paper_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        total_teils = 4 if mod == "lesen" else 2

        yield {
            "type": "init",
            "paper_id": paper_id,
            "module": mod,
            "level": level,
            "total_teils": total_teils,
            "created_at": created_at,
            "duration_minutes": 30,
            "total_points": 25.0
        }

        teils = {}
        aggregated_answer_key = {}
        aggregated_explanations = {}

        if mod == "lesen":
            gen_fns = {
                "teil1": generate_lesen_teil1,
                "teil2": generate_lesen_teil2,
                "teil3": generate_lesen_teil3,
                "teil4": generate_lesen_teil4,
            }
            tasks = [
                (1, "teil1", asyncio.create_task(generate_lesen_teil1(level))),
                (2, "teil2", asyncio.create_task(generate_lesen_teil2(level))),
                (3, "teil3", asyncio.create_task(generate_lesen_teil3(level))),
                (4, "teil4", asyncio.create_task(generate_lesen_teil4(level))),
            ]

            for idx, t_name, gen_task in tasks:
                while not gen_task.done():
                    await asyncio.sleep(2)
                    if not gen_task.done():
                        yield {"type": "ping", "teil_index": idx}

                try:
                    t_data, t_key = await gen_task
                except Exception as e:
                    logger.error("Streaming error in %s: %s", t_name, e)
                    t_data, t_key = await gen_fns[t_name](level)

                teils[t_name] = t_data
                aggregated_answer_key.update(t_key.get("answer_key", {}))
                aggregated_explanations.update(t_key.get("explanations", {}))

                yield {
                    "type": "teil",
                    "teil_index": idx,
                    "teil_name": t_name,
                    "data": t_data
                }

            full_answer_key = {
                "answer_key": aggregated_answer_key,
                "explanations": aggregated_explanations
            }

            paper = ExamPaper(
                paper_id=paper_id,
                module=ExamModule.LESEN,
                level=level,
                created_at=created_at,
                duration_minutes=30,
                total_points=25.0,
                teils=teils,
                answer_key=full_answer_key
            )

        elif mod == "schreiben":
            gen_fns = {
                "teil1": generate_schreiben_teil1,
                "teil2": generate_schreiben_teil2,
            }
            tasks = [
                (1, "teil1", asyncio.create_task(generate_schreiben_teil1(level))),
                (2, "teil2", asyncio.create_task(generate_schreiben_teil2(level))),
            ]

            for idx, t_name, gen_task in tasks:
                while not gen_task.done():
                    await asyncio.sleep(2)
                    if not gen_task.done():
                        yield {"type": "ping", "teil_index": idx}

                try:
                    t_data = await gen_task
                except Exception as e:
                    logger.error("Streaming error in %s: %s", t_name, e)
                    t_data = await gen_fns[t_name](level)

                teils[t_name] = t_data

                yield {
                    "type": "teil",
                    "teil_index": idx,
                    "teil_name": t_name,
                    "data": t_data
                }

            paper = ExamPaper(
                paper_id=paper_id,
                module=ExamModule.SCHREIBEN,
                level=level,
                created_at=created_at,
                duration_minutes=30,
                total_points=25.0,
                teils=teils,
                answer_key=None
            )
        else:
            raise ValueError(f"Unknown exam module: '{module}'. Expected 'lesen' or 'schreiben'.")

        # Persist full paper in Redis
        label = self.storage.store_paper(paper_id, paper.model_dump())

        client_paper = {
            "paper_id": paper.paper_id,
            "label": label,
            "module": paper.module.value,
            "level": paper.level,
            "created_at": paper.created_at,
            "duration_minutes": paper.duration_minutes,
            "total_points": paper.total_points,
            "teils": paper.teils
        }

        yield {
            "type": "done",
            "paper": client_paper
        }

    def evaluate_paper(
        self,
        paper_id: str,
        module: str,
        user_answers: Dict[str, Any],
        level: str = "A2"
    ) -> Dict[str, Any]:
        """Evaluate student exam submission."""
        mod = module.strip().lower()

        # Retrieve paper from Redis
        paper_data = self.storage.get_paper(paper_id)
        if not paper_data:
            logger.warning("Paper %s not found in Redis, using minimal paper wrapper", paper_id)
            paper_data = {
                "paper_id": paper_id,
                "module": mod,
                "level": level,
                "teils": {}
            }

        if mod == "lesen":
            eval_result = evaluate_reading(paper_data, user_answers)
        elif mod == "schreiben":
            eval_result = evaluate_writing(paper_data, user_answers)
        else:
            raise ValueError(f"Unsupported module for evaluation: '{module}'")

        # Save submission evaluation in Redis
        result_dict = eval_result.model_dump()
        self.storage.store_submission(eval_result.submission_id, result_dict)

        # Mark paper completed in Redis
        self.storage.mark_paper_completed(paper_id)

        return result_dict

    def list_saved_papers(
        self,
        module: Optional[str] = None,
        status: Optional[str] = "pending",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Retrieve saved exam question papers available in Redis."""
        return self.storage.list_papers(module=module, status=status, limit=limit)

    def load_saved_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved question paper by ID or label alias, sanitized for client."""
        raw = self.storage.get_paper(paper_id)
        if not raw:
            return None

        raw_mod = raw.get("module", "lesen")
        if isinstance(raw_mod, dict):
            raw_mod = raw_mod.get("value", "lesen")
        mod_str = str(raw_mod).lower().replace("exammodule.", "")

        meta = self.storage.get_paper_meta(paper_id)
        label = raw.get("label") or (meta.get("label") if meta else "") or f"{mod_str.capitalize()} Paper"

        return {
            "paper_id": raw.get("paper_id", paper_id),
            "label": label,
            "module": mod_str,
            "level": raw.get("level", "A2"),
            "created_at": raw.get("created_at", ""),
            "duration_minutes": int(raw.get("duration_minutes", 30)),
            "total_points": float(raw.get("total_points", 25.0)),
            "teils": raw.get("teils", {})
        }

    def delete_saved_paper(self, paper_id: str) -> bool:
        """Delete a saved question paper from Redis."""
        return self.storage.delete_paper(paper_id)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent test submissions summary."""
        return self.storage.get_history(limit=limit)
