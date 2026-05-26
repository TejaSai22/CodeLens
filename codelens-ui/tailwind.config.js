/** @type {import('tailwindcss').Config} */

// Semantic colors are backed by CSS variables (defined in index.css) so the
// same class names work in both light and dark themes. Each variable holds a
// space-separated RGB triple so Tailwind's <alpha-value> opacity modifier works
// (e.g. bg-accent/10).
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        canvas: token("canvas"),     // app background
        panel: token("panel"),       // sidebars / cards
        panel2: token("panel2"),     // raised surfaces
        line: token("border"),       // borders / dividers
        txt: token("text"),          // primary text
        muted: token("muted"),       // secondary text
        faint: token("faint"),       // tertiary text
        accent: token("accent"),     // brand / primary action
        vector: token("vector"),     // semantic (dense) retrieval signal
        keyword: token("keyword"),   // lexical (BM25) retrieval signal
        danger: token("danger"),
        warn: token("warn"),
        ok: token("ok"),
      },
      fontFamily: {
        "inter": ["Inter", "system-ui", "sans-serif"],
        "mono": ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(-100%)", opacity: "0" },
          "20%": { opacity: "1" },
          "80%": { opacity: "1" },
          "100%": { transform: "translateY(2000%)", opacity: "0" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        blink: { "0%,100%": { opacity: "1" }, "50%": { opacity: "0" } },
        "bar-grow": { "0%": { transform: "scaleX(0)" }, "100%": { transform: "scaleX(1)" } },
      },
      animation: {
        scan: "scan 1.1s cubic-bezier(0.4,0,0.2,1) infinite",
        "fade-up": "fade-up 0.35s ease-out both",
        blink: "blink 1s step-end infinite",
        "bar-grow": "bar-grow 0.5s ease-out both",
      },
    },
  },
  plugins: [],
}
