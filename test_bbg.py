"""
Quick Bloomberg test — run this directly to see what your terminal returns.
Usage: python test_bbg.py
"""
import sys

try:
    import blpapi
except ImportError:
    print("ERROR: blpapi not installed. Run: pip install blpapi")
    sys.exit(1)

# ── Connect ──────────────────────────────────────────────────────────────
opts = blpapi.SessionOptions()
opts.setServerHost("localhost")
opts.setServerPort(8194)
session = blpapi.Session(opts)

if not session.start():
    print("ERROR: Cannot start Bloomberg session. Is the terminal running?")
    sys.exit(1)

if not session.openService("//blp/refdata"):
    print("ERROR: Cannot open //blp/refdata service")
    sys.exit(1)

svc = session.getService("//blp/refdata")
print("Connected to Bloomberg.\n")


def bdp_test(tickers, fields):
    """Send a BDP request and print raw results."""
    request = svc.createRequest("ReferenceDataRequest")
    for t in tickers:
        request.append("securities", t)
    for f in fields:
        request.append("fields", f)

    session.sendRequest(request)

    while True:
        event = session.nextEvent(timeout=15000)
        if event.eventType() == blpapi.Event.TIMEOUT:
            print("  TIMEOUT — no response from Bloomberg")
            break

        for msg in event:
            if msg.hasElement("responseError"):
                print(f"  RESPONSE ERROR: {msg}")
                continue
            if not msg.hasElement("securityData"):
                continue

            sec_data = msg.getElement("securityData")
            for i in range(sec_data.numValues()):
                sec = sec_data.getValueAsElement(i)
                ticker = sec.getElementAsString("security")

                if sec.hasElement("securityError"):
                    err = sec.getElement("securityError")
                    cat = err.getElementAsString("category") if err.hasElement("category") else "?"
                    sub = err.getElementAsString("subcategory") if err.hasElement("subcategory") else "?"
                    emsg = err.getElementAsString("message") if err.hasElement("message") else "?"
                    print(f"  {ticker}: SECURITY ERROR [{cat}/{sub}] {emsg}")
                    continue

                if sec.hasElement("fieldExceptions"):
                    fe = sec.getElement("fieldExceptions")
                    for j in range(fe.numValues()):
                        fex = fe.getValueAsElement(j)
                        fid = fex.getElementAsString("fieldId") if fex.hasElement("fieldId") else "?"
                        ei = fex.getElement("errorInfo") if fex.hasElement("errorInfo") else None
                        sub = ei.getElementAsString("subcategory") if ei and ei.hasElement("subcategory") else "?"
                        print(f"  {ticker}: FIELD EXCEPTION for {fid} [{sub}]")

                fd = sec.getElement("fieldData") if sec.hasElement("fieldData") else None
                if fd is None:
                    print(f"  {ticker}: no fieldData")
                    continue

                vals = {}
                for f in fields:
                    if fd.hasElement(f):
                        el = fd.getElement(f)
                        if el.isNull():
                            vals[f] = "NULL"
                        else:
                            try:
                                vals[f] = el.getValueAsFloat()
                            except Exception:
                                try:
                                    vals[f] = el.getValueAsString()
                                except Exception:
                                    vals[f] = f"<type={el.datatype()}>"
                    else:
                        vals[f] = "MISSING"
                print(f"  {ticker}: {vals}")

        if event.eventType() == blpapi.Event.RESPONSE:
            break


# ── TEST 1: FX Spot ──────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: FX SPOT (should work)")
print("=" * 60)
bdp_test(
    ["EURUSD Curncy", "USDJPY Curncy", "GBPUSD Curncy"],
    ["PX_LAST", "PX_BID", "PX_ASK"]
)

# ── TEST 2: FX Vol — ATM ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: FX VOL ATM (1M and 3M)")
print("=" * 60)
bdp_test(
    ["EURUSDV1M Curncy", "EURUSDV3M Curncy", "USDJPYV1M Curncy",
     "USDZARV1M Curncy"],
    ["PX_LAST"]
)

# ── TEST 3: FX Vol — RR and BF ──────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: FX 25D RISK REVERSAL + BUTTERFLY")
print("=" * 60)
bdp_test(
    ["EURUSD25R1M Curncy", "EURUSD25B1M Curncy",
     "EURUSD10R1M Curncy", "EURUSD10B1M Curncy"],
    ["PX_LAST"]
)

# ── TEST 4: Deposit Rates ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: DEPOSIT RATES")
print("=" * 60)
bdp_test(
    ["USDRC Index", "EUDRC Index"],
    ["PX_LAST"]
)

# ── TEST 5: Forward Points ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: FX FORWARD POINTS")
print("=" * 60)
bdp_test(
    ["EUR1M Curncy", "EUR3M Curncy", "JPY1M Curncy"],
    ["PX_LAST"]
)

# ── TEST 6: Try PX_MID as alternative ────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 6: FX VOL with PX_MID (fallback check)")
print("=" * 60)
bdp_test(
    ["EURUSDV1M Curncy", "EURUSDV3M Curncy"],
    ["PX_LAST", "PX_MID", "PX_BID", "PX_ASK"]
)

print("\n" + "=" * 60)
print("DONE. Copy-paste the output above so we can fix the code.")
print("=" * 60)

session.stop()
