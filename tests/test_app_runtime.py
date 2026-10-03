import asyncio
from threading import Event
import unittest

from pianissimo_context.app_runtime import AppRuntime
from pianissimo_context.live import LiveSession


class RuntimeTests(unittest.TestCase):
    def test_start_stop_edit_and_transcript_retention(self):
        entered = Event()
        async def capture(broker, **kwargs):
            # Exercise the real broker/session, with no microphone or models.
            broker.controller = None
            async with LiveSession(broker) as session:
                session.push("Hej")
                broker.emit({"type": "audio_ready"})
                entered.set()
                await kwargs["stop_event"].wait()
        runtime = AppRuntime(capture)
        try:
            runtime.start(2, "local", "Initial bakgrund")
            self.assertTrue(entered.wait(3))
            self.assertEqual(runtime.snapshot().transcript, "Hej")
            self.assertEqual(runtime.snapshot().status, "Lyssnar")
            with self.assertRaises(ValueError):
                runtime.start(2, "local", "Dubbel start")
            runtime.update_context("Ny kontext med rättat namn")
            snapshot = runtime.snapshot()
            self.assertEqual(snapshot.context, "Ny kontext med rättat namn")
            self.assertGreater(snapshot.revision, 0)
            runtime.stop()
            runtime._future.result(timeout=3)
            self.assertFalse(runtime.snapshot().running)
            self.assertEqual(runtime.snapshot().transcript, "Hej")
            with self.assertRaises(ValueError):
                runtime.update_context("Efter stopp")
        finally:
            runtime.shutdown()

    def test_validation_before_capture(self):
        runtime = AppRuntime()
        try:
            for device, model, context in [(None, "model", ""), ("bad", "model", ""),
                                           (-1, "model", ""), (True, "model", ""),
                                           (0, "", ""), (0, "model", "x" * 4001)]:
                with self.assertRaises(ValueError):
                    runtime.start(device, model, context)
            with self.assertRaises(ValueError):
                runtime.update_context("")
            self.assertFalse(runtime.snapshot().running)
        finally:
            runtime.shutdown()

    def test_restart_waits_for_audio_drain_and_context_versions_do_not_repeat(self):
        entered, draining, release = Event(), Event(), Event()
        async def capture(broker, **kwargs):
            entered.set()
            await kwargs["stop_event"].wait()
            draining.set()
            await asyncio.to_thread(release.wait, 3)
        runtime = AppRuntime(capture)
        try:
            runtime.start(0, "model", "Första")
            self.assertTrue(entered.wait(2))
            first_revision = runtime.snapshot().revision
            runtime.stop()
            self.assertTrue(draining.wait(2))
            self.assertTrue(runtime.snapshot().running)
            self.assertEqual(runtime.snapshot().status, "Stoppar")
            with self.assertRaises(ValueError):
                runtime.start(0, "model", "För tidigt")
            release.set()
            runtime._future.result(timeout=3)
            runtime.start(0, "model", "Andra")
            self.assertGreater(runtime.snapshot().revision, first_revision)
        finally:
            release.set()
            runtime.shutdown()
        with self.assertRaises(ValueError):
            runtime.update_context("För sent")

    def test_failure_unlocks_controls_without_logging_sensitive_message(self):
        async def fail(broker, **kwargs):
            raise RuntimeError("sensitive transcript")
        runtime = AppRuntime(fail)
        try:
            runtime.start(0, "model", "")
            runtime._future.result(timeout=3)
            s = runtime.snapshot()
            self.assertFalse(s.running)
            self.assertEqual(s.message, "Kunde inte starta (RuntimeError)")
        finally:
            runtime.shutdown()
