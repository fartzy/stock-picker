import { useEffect, useRef, useState } from "react";
import type { HistoryDefaults } from "../appSettings";
import { HISTORY_GROUPINGS } from "../tradeHistoryPeriods";

type SettingsProps = {
  defaults: HistoryDefaults;
  onChange: (defaults: HistoryDefaults) => void;
  saved: boolean;
};

function SettingsDialog({ defaults, onChange, saved, onClose }: SettingsProps & { onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [section, setSection] = useState<"settings" | "defaults">("settings");
  const headingRef = useRef<HTMLHeadingElement>(null);
  const defaultsButtonRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const dialog = dialogRef.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  useEffect(() => {
    if (section === "defaults") headingRef.current?.focus();
    else defaultsButtonRef.current?.focus();
  }, [section]);

  return (
    <dialog
      ref={dialogRef}
      className="settings-dialog"
      aria-labelledby="settings-title"
      onCancel={(event) => { event.preventDefault(); onClose(); }}
      onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}
    >
      <div className="settings-dialog-inner">
        <div className="settings-dialog-heading">
          <div>
            {section === "defaults" && <button type="button" className="settings-back" onClick={() => {
              setSection("settings");
            }}>← Settings</button>}
            <h2 ref={headingRef} tabIndex={-1} id="settings-title">{section === "settings" ? "Settings" : "Defaults"}</h2>
          </div>
          <button type="button" className="settings-icon-button" aria-label="Close settings" onClick={onClose}>×</button>
        </div>
        {section === "settings" ? (
          <button ref={defaultsButtonRef} type="button" className="settings-category" onClick={() => setSection("defaults")}>
            <strong>Defaults</strong>
          </button>
        ) : (
          <>
          <section className="settings-default-row" aria-labelledby="history-view-label">
            <div>
              <h3 id="history-view-label">Default view</h3>
            </div>
            <div className="settings-checkboxes" role="radiogroup" aria-labelledby="history-view-label">
              {HISTORY_GROUPINGS.map((option) => (
                <label key={option.value}>
                  <input type="radio" name="default-history-view" value={option.value}
                    checked={defaults.grouping === option.value}
                    onChange={() => onChange({ ...defaults, grouping: option.value })} />
                  {option.label}
                </label>
              ))}
            </div>
          </section>
          <section className="settings-default-row" aria-labelledby="history-defaults-label">
            <div>
              <h3 id="history-defaults-label">Keep most recent open</h3>
            </div>
            <div className="settings-checkboxes" role="group" aria-labelledby="history-defaults-label">
              <label>
                <input type="checkbox" checked={defaults.keepRecentWeekOpen}
                  onChange={(event) => onChange({ ...defaults, keepRecentWeekOpen: event.currentTarget.checked })} />
                Week
              </label>
              <label>
                <input type="checkbox" checked={defaults.keepRecentMonthOpen}
                  onChange={(event) => onChange({ ...defaults, keepRecentMonthOpen: event.currentTarget.checked })} />
                Month
              </label>
            </div>
          </section>
          </>
        )}
        <div className="settings-dialog-footer">
          <p role="status">{!saved ? "Applied for now. Browser settings could not be saved."
            : section === "defaults" ? "Saved automatically in this browser." : "Preferences for this browser."}</p>
          <button type="button" className="settings-done" onClick={onClose}>Done</button>
        </div>
      </div>
    </dialog>
  );
}

export default function AppSettings(props: SettingsProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const settingsRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    settingsRef.current?.focus();
    const dismissOutside = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    const dismissWithEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMenuOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", dismissOutside);
    document.addEventListener("keydown", dismissWithEscape);
    return () => {
      document.removeEventListener("pointerdown", dismissOutside);
      document.removeEventListener("keydown", dismissWithEscape);
    };
  }, [menuOpen]);

  return (
    <div className="app-settings" ref={menuRef} onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) setMenuOpen(false);
    }}>
      <button ref={triggerRef} type="button" className="settings-icon-button"
        aria-label="App settings" aria-expanded={menuOpen} aria-controls="app-settings-menu"
        title="App settings" onClick={() => setMenuOpen((open) => !open)}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="m9 3-.6 2.4-1.7 1L4.3 6l-2 3.5L4 11.2v1.9l-1.7 1.7 2 3.5 2.4-.7 1.7 1L9 21h4l.6-2.4 1.7-1 2.4.7 2-3.5-1.7-1.7v-1.9l1.7-1.7-2-3.5-2.4.4-1.7-1L13 3Z" />
          <circle cx="11" cy="12" r="3" />
        </svg>
      </button>
      {menuOpen && (
        <div id="app-settings-menu" className="settings-menu">
          <button ref={settingsRef} type="button" onClick={() => { setMenuOpen(false); setDialogOpen(true); }}>
            Settings
          </button>
        </div>
      )}
      {dialogOpen && <SettingsDialog {...props} onClose={() => {
        setDialogOpen(false);
        triggerRef.current?.focus();
      }} />}
    </div>
  );
}
