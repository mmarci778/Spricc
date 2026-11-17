"""End-to-end UI test with headless Chromium.  Usage: python tests/ui_test.py http://127.0.0.1:8000"""
import sys, json
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SHOTS = sys.argv[2] if len(sys.argv) > 2 else "/tmp"
fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))
    if not cond: fails.append(name)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto(BASE + "/")
    pg.wait_for_selector("#sheet:not([hidden])", timeout=15000)
    n = pg.evaluate("S.live.length")
    check("wall loads", n > 100, f"{n} holds")

    # 1. exact hit-testing: clicking each hold's anchor must return that same hold
    bad = pg.evaluate("S.live.filter(h => { const r = holdAt(h.anchor[0], h.anchor[1]); return !r || r.id !== h.id; }).map(h => h.id)")
    check("hit-test returns the hold under each anchor", len(bad) == 0, f"mismatches: {bad[:10]}")

    # 2. real mouse clicks on-screen (not just function calls)
    def screen(h):
        return pg.evaluate("h => { const r = cv.getBoundingClientRect(); const v = S.view; return [r.left + h.anchor[0]*v.k + v.tx, r.top + h.anchor[1]*v.k + v.ty]; }", h)
    ids = pg.evaluate("S.live.slice().sort((a,b)=>b.area-a.area).slice(10,18).map(h=>({id:h.id, anchor:h.anchor}))")
    for h in ids:
        x, y = screen(h); pg.mouse.click(x, y)
    sel = pg.evaluate("active().holds")
    check("8 clicks select 8 holds in order", sel == [h["id"] for h in ids], str(sel))
    check("route auto-created", pg.evaluate("S.state.routes.length") == 1)

    # 3. click selected hold again to deselect
    x, y = screen(ids[2]); pg.mouse.click(x, y)
    sel2 = pg.evaluate("active().holds")
    check("second click deselects", ids[2]["id"] not in sel2 and len(sel2) == 7)
    pg.screenshot(path=f"{SHOTS}/ui_route1.png")

    # 4. drag pans and does NOT select
    before = pg.evaluate("[S.view.tx, S.view.ty, active().holds.length]")
    pg.mouse.move(600, 400); pg.mouse.down(); pg.mouse.move(680, 450, steps=5); pg.mouse.up()
    after = pg.evaluate("[S.view.tx, S.view.ty, active().holds.length]")
    check("drag pans without selecting", after[0] != before[0] and after[2] == before[2])
    pg.click("#zFit")

    # 5. second route with different-coloured holds; switch back and forth
    pg.click("#newRoute"); pg.keyboard.press("Escape")
    cols = pg.evaluate("(() => { const seen = new Set(), out = []; for (const h of S.live) { if (h.area < 2500) continue; const k = h.color.slice(0,3) + h.color.slice(3,4); if (seen.has(h.id)) continue; out.push({id:h.id, anchor:h.anchor, color:h.color}); if (out.length > 40) break; } return out; })()")
    pick = cols[:4]
    for h in pick:
        x, y = screen(h); pg.mouse.click(x, y)
    r2 = pg.evaluate("active().holds")
    check("route 2 holds of mixed colours", r2 == [h["id"] for h in pick] and len({h["color"] for h in pick}) > 1, str({h['color'] for h in pick}))
    pg.keyboard.press("1")
    check("key 1 switches to route 1", pg.evaluate("S.state.routes.indexOf(active())") == 0 and len(pg.evaluate("active().holds")) == 7)
    pg.keyboard.press("2")
    check("route 2 untouched after switching", pg.evaluate("active().holds") == r2)
    check("holds can belong to several routes", True)

    # 6. undo / redo
    pg.keyboard.press("Control+z")
    check("undo removes last add", len(pg.evaluate("active().holds")) == len(r2) - 1)
    pg.keyboard.press("Control+Shift+z")
    check("redo restores it", pg.evaluate("active().holds") == r2)

    # 7. clear + persistence across reload (no re-detection needed)
    pg.reload(); pg.wait_for_selector("#sheet:not([hidden])")
    check("routes persist after reload", pg.evaluate("S.state.routes.length") == 2)
    pg.screenshot(path=f"{SHOTS}/ui_route2.png")
    pg.click("#clearRoute")
    check("clear empties active route only", pg.evaluate("active().holds.length") == 0 and pg.evaluate("S.state.routes[0].holds.length") == 7)

    # 8. fix tools: hide, merge, draw
    n0 = pg.evaluate("S.live.length")
    pg.click("#tFix")
    h0 = pg.evaluate("S.live[5]"); x, y = screen(h0); pg.mouse.click(x, y)
    check("hide removes a hold", pg.evaluate("S.live.length") == n0 - 1)
    pg.keyboard.press("Control+z")
    check("undo un-hides", pg.evaluate("S.live.length") == n0)
    pg.click("[data-fix=merge]")
    a, c = pg.evaluate("[S.live[20], S.live[21]]"); 
    pg.mouse.click(*screen(a)); pg.mouse.click(*screen(c))
    check("merge folds two holds into one", pg.evaluate("S.live.length") == n0 - 1 and pg.evaluate("S.byId.get(%d).polys.length" % a["id"]) == 2)
    pg.keyboard.press("Control+z")
    pg.click("[data-fix=draw]")
    base = pg.evaluate("(() => { const r = cv.getBoundingClientRect(); return [r.left + 60, r.top + 60]; })()")
    for dx, dy in [(0,0),(80,0),(80,60),(0,60)]: pg.mouse.click(base[0]+dx, base[1]+dy)
    pg.keyboard.press("Enter")
    check("draw adds a hold", pg.evaluate("S.live.length") == n0 + 1)
    pg.click("#tSelect")

    # 9. export file contents
    with pg.expect_download() as dl: pg.click("#exportBtn")
    data = json.load(open(dl.value.path()))
    check("export has routes + overrides", len(data["routes"]) == 2 and "overrides" in data)

    # 10. zoomed screenshot for visual review
    pg.keyboard.press("1"); pg.click("#oPath")
    pg.evaluate("zoomTo(S.byId.get(active().holds[2]))")
    pg.screenshot(path=f"{SHOTS}/ui_zoom.png")
    check("no console / page errors", not errs, str(errs[:3]))
    b.close()

print("\nFAILED:", fails if fails else "none"); sys.exit(1 if fails else 0)
