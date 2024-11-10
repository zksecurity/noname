// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="intro.html"><strong aria-hidden="true">1.</strong> Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Language</li><li class="chapter-item expanded "><a href="basics.html"><strong aria-hidden="true">2.</strong> Basics</a></li><li class="chapter-item expanded affix "><li class="part-title">Compiler Internals</li><li class="chapter-item expanded "><a href="compilation.html"><strong aria-hidden="true">3.</strong> Compilation</a></li><li class="chapter-item expanded "><a href="grammar.html"><strong aria-hidden="true">4.</strong> Grammar</a></li><li class="chapter-item expanded "><a href="spans.html"><strong aria-hidden="true">5.</strong> Spans</a></li><li class="chapter-item expanded "><a href="paths.html"><strong aria-hidden="true">6.</strong> Paths</a></li><li class="chapter-item expanded "><a href="type-checker.html"><strong aria-hidden="true">7.</strong> Type Checker</a></li><li class="chapter-item expanded "><a href="asm.html"><strong aria-hidden="true">8.</strong> Noname ASM</a></li><li class="chapter-item expanded "><a href="structs.html"><strong aria-hidden="true">9.</strong> Structs</a></li><li class="chapter-item expanded "><a href="methods.html"><strong aria-hidden="true">10.</strong> Methods</a></li><li class="chapter-item expanded "><a href="literals.html"><strong aria-hidden="true">11.</strong> Literals and the const keyword</a></li><li class="chapter-item expanded "><a href="expressions.html"><strong aria-hidden="true">12.</strong> Expressions</a></li><li class="chapter-item expanded "><a href="modules.html"><strong aria-hidden="true">13.</strong> Modules</a></li><li class="chapter-item expanded affix "><li class="part-title">Circuit Generation</li><li class="chapter-item expanded "><a href="cellvar.html"><strong aria-hidden="true">14.</strong> CellVars</a></li><li class="chapter-item expanded "><a href="var.html"><strong aria-hidden="true">15.</strong> Vars</a></li><li class="chapter-item expanded "><a href="constants.html"><strong aria-hidden="true">16.</strong> Constants</a></li><li class="chapter-item expanded "><a href="functions.html"><strong aria-hidden="true">17.</strong> Functions</a></li><li class="chapter-item expanded "><a href="scope.html"><strong aria-hidden="true">18.</strong> Scope</a></li><li class="chapter-item expanded affix "><li class="part-title">Proof Creation</li><li class="chapter-item expanded "><a href="public-outputs.html"><strong aria-hidden="true">19.</strong> Public Outputs</a></li><li class="chapter-item expanded "><a href="witness-generation.html"><strong aria-hidden="true">20.</strong> Witness Generation</a></li><li class="chapter-item expanded affix "><li class="part-title">RFCs</li><li class="chapter-item expanded "><a href="rfc/rfc-0-generic-parameters.html"><strong aria-hidden="true">21.</strong> RFC-0 Generic Parameters</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
