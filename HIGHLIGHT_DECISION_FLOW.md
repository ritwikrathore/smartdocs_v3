# Highlight Method Decision Flow

The verifier now keeps two separate buffers while scanning chunks for the same phrase:

- **Exact buffer** – precise rectangles returned by `page.search_for()`
- **Fallback buffer** – chunk/page rectangles obtained from fuzzy matches, cross-page stitching, or quote heuristics

Only after every chunk has been examined do we decide which buffer to commit:

1. If the exact buffer contains any rectangles, we keep only those and discard the fallback buffer.
2. Otherwise we fall back to the chunk/page rectangles.
3. Statistics are updated from the committed list, so counts always reflect the highlights actually applied to the PDF.

## Walkthrough

1. Initialise `phrase_exact_locations = []` and `phrase_fallback_locations = []`.
2. Walk chunks that cleared the fuzzy-score threshold.
      - When `page.search_for()` succeeds, append rectangles to the exact buffer.
      - When search fails but fuzzy detection passes, append rectangles to the fallback buffer.
      - Cross-page and quote-handling heuristics also append to the fallback buffer.
3. After all chunks are processed:
      - Commit exact buffer if non-empty; otherwise commit fallback buffer.
      - Record highlight statistics from the committed rectangles only.

The end result is that a phrase can never receive both exact and fallback highlights simultaneously, and the statistics logged for each run match the annotations embedded in the output PDF.
