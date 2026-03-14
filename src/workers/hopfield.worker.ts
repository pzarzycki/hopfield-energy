/// <reference lib="webworker" />

import { HopfieldEngine } from "../core/hopfield";
import { createBlankPattern, getPatternSetById } from "../core/patternSets";
import type { WorkerRequest, WorkerResponse } from "../core/workerProtocol";

const engine = new HopfieldEngine();

let initialState = createBlankPattern();
let playTimer: number | null = null;

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer);
}

function pausePlayback(): void {
  if (playTimer !== null) {
    self.clearInterval(playTimer);
    playTimer = null;
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  try {
    if (message.type === "initialize") {
      pausePlayback();

      const patternSet = getPatternSetById(message.patternSetId);
      const init = engine.train(patternSet.patterns, message.updateRule);
      initialState = createBlankPattern();
      const snapshot = engine.setState(initialState);

      post(
        {
          type: "ready",
          ...init,
          snapshot,
        },
        [init.weights.buffer, snapshot.state.buffer],
      );
      return;
    }

    if (message.type === "setQuery") {
      pausePlayback();
      initialState = message.pattern.slice();
      const snapshot = engine.setState(initialState);
      post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);
      return;
    }

    if (message.type === "reset") {
      pausePlayback();
      const snapshot = engine.reset(initialState);
      post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);
      return;
    }

    if (message.type === "step") {
      pausePlayback();
      const snapshot = engine.step();
      post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);
      return;
    }

    if (message.type === "play") {
      pausePlayback();

      let iteration = 0;
      playTimer = self.setInterval(() => {
        const snapshot = engine.step();
        post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);

        iteration += 1;
        if (snapshot.converged || iteration >= message.maxSteps) {
          pausePlayback();
          post({ type: "paused" });
        }
      }, message.intervalMs);
      return;
    }

    if (message.type === "pause") {
      pausePlayback();
      post({ type: "paused" });
    }
  } catch (error) {
    pausePlayback();
    post({
      type: "error",
      message: error instanceof Error ? error.message : "Unknown worker error",
    });
  }
};
