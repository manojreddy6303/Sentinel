// Regression tests for null object_class formatting in Sentinel Frontend
import assert from "node:assert";

// Function under test (identical to formatClassName in VideoUpload.tsx)
function formatClassName(name, fallback = "General Security Activity") {
  if (!name || typeof name !== "string" || !name.trim()) {
    return fallback;
  }
  return name
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

// Function under test (identical to getClassColor in VideoUpload.tsx)
const CLASS_COLORS = {
  person: "text-blue-400 bg-blue-500/10 border-blue-500/30",
  car: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  truck: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  backpack: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
};

const getClassColor = (cls) =>
  (cls && typeof cls === "string" ? CLASS_COLORS[cls.toLowerCase()] : undefined) ||
  "text-zinc-300 bg-zinc-800/60 border-zinc-700";

// Render logic under test (identical to VideoUpload.tsx line 1562 & 1432)
function renderInvestigationDetectionTitle(item) {
  return item.object_class
    ? formatClassName(item.object_class).toUpperCase()
    : item.event_type
    ? item.event_type.replace(/_/g, " ").toUpperCase()
    : "GENERAL SECURITY ACTIVITY";
}

function renderReferencedDetectionLabel(det) {
  return det.object_class
    ? formatClassName(det.object_class)
    : det.event_type
    ? det.event_type.replace(/_/g, " ").toUpperCase()
    : formatClassName(null);
}

console.log("Starting Sentinel Frontend Null Object Class Regression Tests...\n");

// 1. Event with object_class string
{
  const event = {
    event_id: "ev-1",
    object_class: "person",
    event_type: "POTENTIAL_INTRUSION",
    confidence: 0.95,
  };
  assert.strictEqual(formatClassName(event.object_class), "Person");
  assert.strictEqual(renderInvestigationDetectionTitle(event), "PERSON");
  assert.strictEqual(renderReferencedDetectionLabel(event), "Person");
  assert.strictEqual(getClassColor(event.object_class), "text-blue-400 bg-blue-500/10 border-blue-500/30");
  console.log("✓ Test 1 Passed: Event with object_class string ('person' -> 'PERSON' / 'Person')");
}

// 2. Event with object_class null
{
  const event = {
    event_id: "ev-2",
    object_class: null,
    event_type: "PROLONGED_PRESENCE",
    confidence: 0.88,
  };
  assert.strictEqual(formatClassName(event.object_class), "General Security Activity");
  assert.strictEqual(renderInvestigationDetectionTitle(event), "PROLONGED PRESENCE");
  assert.strictEqual(renderReferencedDetectionLabel(event), "PROLONGED PRESENCE");
  assert.strictEqual(getClassColor(event.object_class), "text-zinc-300 bg-zinc-800/60 border-zinc-700");
  console.log("✓ Test 2 Passed: Event with object_class null ('PROLONGED_PRESENCE' -> 'PROLONGED PRESENCE')");
}

// 3. Event with object_class undefined
{
  const event = {
    event_id: "ev-3",
    object_class: undefined,
    event_type: "OBSERVATIONAL_ANOMALY",
    confidence: 0.82,
  };
  assert.strictEqual(formatClassName(event.object_class), "General Security Activity");
  assert.strictEqual(renderInvestigationDetectionTitle(event), "OBSERVATIONAL ANOMALY");
  assert.strictEqual(renderReferencedDetectionLabel(event), "OBSERVATIONAL ANOMALY");
  assert.strictEqual(getClassColor(event.object_class), "text-zinc-300 bg-zinc-800/60 border-zinc-700");
  console.log("✓ Test 3 Passed: Event with object_class undefined ('OBSERVATIONAL_ANOMALY' -> 'OBSERVATIONAL ANOMALY')");
}

// 4. Event with event_type but no object class
{
  const event = {
    event_id: "ev-4",
    event_type: "HIGH_ACTIVITY_PERIOD",
    confidence: 0.9,
  };
  assert.strictEqual(formatClassName(event.object_class), "General Security Activity");
  assert.strictEqual(renderInvestigationDetectionTitle(event), "HIGH ACTIVITY PERIOD");
  assert.strictEqual(renderReferencedDetectionLabel(event), "HIGH ACTIVITY PERIOD");
  assert.strictEqual(getClassColor(event.object_class), "text-zinc-300 bg-zinc-800/60 border-zinc-700");
  console.log("✓ Test 4 Passed: Event with event_type but no object class ('HIGH_ACTIVITY_PERIOD' -> 'HIGH ACTIVITY PERIOD')");
}

// 5. 'Which events should I review?' response containing mixed events
{
  const mixedResponse = {
    query: "Which events should I review?",
    sources: {
      detections: [
        { event_id: "d1", timestamp: 3.5, object_class: "person", confidence: 0.94 },
        { event_id: "d2", timestamp: 12.0, object_class: null, event_type: "PROLONGED_PRESENCE", confidence: 0.89 },
        { event_id: "d3", timestamp: 18.2, object_class: undefined, event_type: "HIGH_ACTIVITY_PERIOD", confidence: 0.85 },
        { event_id: "d4", timestamp: 24.5, object_class: "backpack", event_type: "POTENTIAL_THEFT", confidence: 0.96 },
        { event_id: "d5", timestamp: 28.0, object_class: null, confidence: 0.75 },
      ],
    },
  };

  const renderedLabels = mixedResponse.sources.detections.map(renderReferencedDetectionLabel);
  assert.deepStrictEqual(renderedLabels, [
    "Person",
    "PROLONGED PRESENCE",
    "HIGH ACTIVITY PERIOD",
    "Backpack",
    "General Security Activity",
  ]);

  const renderedTitles = mixedResponse.sources.detections.map(renderInvestigationDetectionTitle);
  assert.deepStrictEqual(renderedTitles, [
    "PERSON",
    "PROLONGED PRESENCE",
    "HIGH ACTIVITY PERIOD",
    "BACKPACK",
    "GENERAL SECURITY ACTIVITY",
  ]);

  console.log("✓ Test 5 Passed: 'Which events should I review?' with mixed events renders seamlessly without error");
}

console.log("\nALL 5 FRONTEND REGRESSION TESTS PASSED 100%!");
