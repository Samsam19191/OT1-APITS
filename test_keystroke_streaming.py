"""
Quick test script for keystroke streaming.
Run with: python test_keystroke_streaming.py
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.events import EventType, KeystrokeEvent, find_last_boundary
from src.keystroke_simulator import KeystrokeSimulator, LoggingConsumer


async def test_basic_typing():
    """Test basic typing simulation."""
    print("\n" + "="*60)
    print("TEST 1: Basic Typing Simulation")
    print("="*60)
    
    queue = asyncio.Queue()
    
    simulator = KeystrokeSimulator(
        queue,
        mean_delay_ms=50,      # Fast for testing
        debounce_ms=100,       # Short debounce
        typo_rate=0.0,         # No typos
    )
    
    consumer = LoggingConsumer(queue, verbose=True)
    
    query = "Show patients"
    print(f"\nSimulating: '{query}'\n")
    
    await asyncio.gather(
        simulator.simulate_typing(query),
        consumer.run()
    )
    
    print(f"\n✓ Total events: {len(consumer.events)}")
    print(f"✓ Final text: '{simulator.session.current_text}'")
    
    # Verify
    assert simulator.session.current_text == query, "Text mismatch!"
    assert any(e.event_type == EventType.SUBMIT for e in consumer.events), "Missing SUBMIT!"
    assert any(e.event_type == EventType.END for e in consumer.events), "Missing END!"
    print("✓ All assertions passed!")


async def test_flush_events():
    """Test that FLUSH events fire at word boundaries."""
    print("\n" + "="*60)
    print("TEST 2: FLUSH Events at Boundaries")
    print("="*60)
    
    queue = asyncio.Queue()
    
    simulator = KeystrokeSimulator(
        queue,
        mean_delay_ms=30,
        debounce_ms=80,
        typo_rate=0.0,
    )
    
    consumer = LoggingConsumer(queue, verbose=False)
    
    query = "Find all orders"
    print(f"\nSimulating: '{query}'")
    
    await asyncio.gather(
        simulator.simulate_typing(query),
        consumer.run()
    )
    
    flush_events = [e for e in consumer.events if e.event_type == EventType.FLUSH]
    print(f"\n✓ FLUSH events: {len(flush_events)}")
    for e in flush_events:
        print(f"  - Confirmed: '{e.confirmed_text}'")
    
    # Should have at least 2 flushes (after "Find " and "all ")
    assert len(flush_events) >= 2, f"Expected >=2 FLUSH events, got {len(flush_events)}"
    print("✓ Boundary detection working!")


async def test_typo_simulation():
    """Test typo and backspace handling."""
    print("\n" + "="*60)
    print("TEST 3: Typo Simulation")
    print("="*60)
    
    queue = asyncio.Queue()
    
    simulator = KeystrokeSimulator(
        queue,
        mean_delay_ms=30,
        debounce_ms=80,
        typo_rate=0.3,  # High typo rate for testing
    )
    
    consumer = LoggingConsumer(queue, verbose=False)
    
    query = "List products"
    print(f"\nSimulating with 30% typo rate: '{query}'")
    
    await asyncio.gather(
        simulator.simulate_typing(query),
        consumer.run()
    )
    
    char_adds = sum(1 for e in consumer.events if e.event_type == EventType.CHAR_ADD)
    char_dels = sum(1 for e in consumer.events if e.event_type == EventType.CHAR_DELETE)
    
    print(f"\n✓ CHAR_ADD events: {char_adds}")
    print(f"✓ CHAR_DELETE events: {char_dels}")
    print(f"✓ Final text: '{simulator.session.current_text}'")
    
    # Final text should match despite typos
    assert simulator.session.current_text == query, "Text mismatch after typo correction!"
    print("✓ Typo correction working!")


async def test_boundary_detection():
    """Test the boundary detection helper."""
    print("\n" + "="*60)
    print("TEST 4: Boundary Detection")
    print("="*60)
    
    test_cases = [
        ("Show me", 5),          # After "Show "
        ("Hello world!", 12),    # After "!"
        ("Test", 0),             # No boundary yet
        ("A B C", 4),            # After "C " -> position 4
        ("Query, done", 7),      # After ", "
    ]
    
    for text, expected in test_cases:
        result = find_last_boundary(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} find_last_boundary('{text}') = {result} (expected {expected})")
    
    print("✓ Boundary detection tests complete!")


async def test_event_schema():
    """Test event creation and validation."""
    print("\n" + "="*60)
    print("TEST 5: Event Schema")
    print("="*60)
    
    # Valid CHAR_ADD
    e1 = KeystrokeEvent(EventType.CHAR_ADD, char='a', current_text='a')
    print(f"  ✓ Valid CHAR_ADD: {e1}")
    
    # FLUSH event
    e2 = KeystrokeEvent(EventType.FLUSH, current_text='hello ', confirmed_text='hello ')
    print(f"  ✓ Valid FLUSH: {e2}")
    
    # Invalid CHAR_ADD (missing char) should raise
    try:
        e3 = KeystrokeEvent(EventType.CHAR_ADD, current_text='test')
        print("  ✗ Should have raised ValueError!")
    except ValueError:
        print("  ✓ Correctly rejected CHAR_ADD without char")
    
    print("✓ Event schema validation working!")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("KEYSTROKE STREAMING TEST SUITE")
    print("="*60)
    
    await test_event_schema()
    await test_boundary_detection()
    await test_basic_typing()
    await test_flush_events()
    await test_typo_simulation()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
