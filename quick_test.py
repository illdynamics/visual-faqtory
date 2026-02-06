#!/usr/bin/env python3
"""
quick_test.py - Quick smoke test for Visual FaQtory v0.0.7-alpha
"""
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    try:
        from vfaq import VisualFaQtory

        worqspace_dir = Path('./worqspace').resolve()
        output_dir = Path('./qodeyard').resolve()

        output_dir.mkdir(parents=True, exist_ok=True)

        config_override = {'backend': {'type': 'mock'}}

        logger.info("QonQrete Visual FaQtory v0.0.7-alpha - Quick Test")
        logger.info("=" * 50)
        logger.info("Initializing with Mock Backend...")

        faqtory = VisualFaQtory(
            worqspace_dir=worqspace_dir,
            output_dir=output_dir,
            config_override=config_override
        )

        logger.info("Running single test cycle...")
        briq = faqtory.run_single_cycle(cycle_index=0)

        logger.info(f"Cycle complete!")
        logger.info(f"  Briq ID: {briq.briq_id}")
        logger.info(f"  Status: {briq.status.value}")
        if briq.looped_video_path:
            logger.info(f"  Looped video: {briq.looped_video_path}")
        else:
            logger.info(f"  Looped video: Not generated (check backend/FFmpeg)")
        logger.info(f"  Evolution: {briq.evolution_suggestion}")
        logger.info("Quick test finished. Check the output directory.")

    except ImportError:
        logger.error("Error: 'vfaq' module not found. "
                     "Ensure you are in the project root and have installed dependencies.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    main()
