import { useEffect, useState } from "react";

import {
  LOCAL_PREFS_CHANGED_EVENT,
  readLocalPreferences,
  type LocalPreferences,
} from "@/lib/local-preferences";

export function useShowModelRouting(): boolean {
  const [enabled, setEnabled] = useState(() => readLocalPreferences().showModelRouting);

  useEffect(() => {
    const refresh = () => setEnabled(readLocalPreferences().showModelRouting);
    const refreshFromLocalPreferenceEvent = (event: Event) => {
      const detail = (event as CustomEvent<Partial<LocalPreferences> | undefined>).detail;
      setEnabled(
        detail?.showModelRouting !== undefined
          ? detail.showModelRouting
          : readLocalPreferences().showModelRouting,
      );
    };
    window.addEventListener("storage", refresh);
    window.addEventListener("focus", refresh);
    window.addEventListener(LOCAL_PREFS_CHANGED_EVENT, refreshFromLocalPreferenceEvent);
    return () => {
      window.removeEventListener("storage", refresh);
      window.removeEventListener("focus", refresh);
      window.removeEventListener(LOCAL_PREFS_CHANGED_EVENT, refreshFromLocalPreferenceEvent);
    };
  }, []);

  return enabled;
}
