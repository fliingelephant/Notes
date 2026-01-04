This repository organizes notes by topic in separate folders. This root `README.md` serves as a guideline for composing, organizing, and modifying notes.

For each run, create a folder-specific `AGENTS.md` and delete it afterward. Without explicit user instructions, do not read or index files from folders other than the current one, unless for syntax reference.

When compiling to PDF, use a suffix indicating the source type (e.g., `{topic}.tex` → `{topic}_latex.pdf`, `{topic}.typ` → `{topic}_typst.pdf`, `{topic}.md` → `{topic}_markdown.pdf`).

## Typst 

### Composition

- For Typst syntax and functions, refer to the [Typst Reference](https://typst.app/docs/reference).
- Always use `physica` for scientific notation. Refer to the [Physica documentation](https://typst.app/universe/package/physica/) and [Physica manual](https://github.com/Leedehai/typst-physics/blob/master/physica-manual.pdf) for available functions (derivatives, bra-ket, tensors, etc.).
- For plotting and charts, refer to the [CeTZ documentation](https://typst.app/universe/package/cetz/) and [CeTZ manual](https://cetz-package.github.io/docs).

### Compilation

```bash
typst compile {topic}.typ {topic}_typst.pdf
typst watch {topic}.typ {topic}_typst.pdf  # auto-recompile on save
```
