//! Lock-free triple buffer for single-producer / single-consumer hand-off of a
//! `Copy` snapshot (here: [`crate::Features`]).
//!
//! The DSP worker is the *producer*: it writes a fresh snapshot into its private
//! "back" slot, then publishes it with a single atomic swap. The render/UI
//! thread is the *consumer*: [`Reader::read`] never blocks and always returns the
//! most recently published snapshot. Stale reads are impossible to tear because
//! publication is a single atomic store of a slot index, and each slot is owned
//! by exactly one side at a time.
//!
//! Layout: three slots + one atomic control word. The control word packs:
//!   - `back`  (bits 0..2): slot the producer is currently writing into.
//!   - `ready` (bits 2..4): most-recently-published slot (+ a dirty flag in bit 4
//!     so the consumer can tell "new data since my last read", though we don't
//!     strictly need it for our always-read-latest use).
//!
//! Implementation uses the well-known "swap back and ready" scheme: the producer
//! atomically swaps its `back` index with the published `ready` index, so the two
//! sides never touch the same slot concurrently.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// Shared state behind the triple buffer. Three storage slots and one control byte.
struct Shared<T> {
    slots: [UnsafeCell<T>; 3],
    /// Bits 0..2: index (0..=2) of the slot the consumer should read next.
    /// Bit 2 (value 4): "dirty" flag — set by producer on publish, cleared by reader.
    ready: AtomicU8,
}

// SAFETY: access to each slot is disciplined by the `ready` control byte. At any
// instant the producer owns its private `back` slot and the consumer owns the
// slot it last latched; the third slot is the published one. The atomic swap of
// indices is the synchronization edge, so no two threads ever alias a slot.
unsafe impl<T: Send> Sync for Shared<T> {}
unsafe impl<T: Send> Send for Shared<T> {}

const INDEX_MASK: u8 = 0b011;
const DIRTY_BIT: u8 = 0b100;

/// Producer half. Lives on the DSP worker thread.
pub struct Writer<T> {
    shared: Arc<Shared<T>>,
    /// Slot the producer currently owns and writes into.
    back: u8,
}

/// Consumer half. Lives on the render/UI thread.
///
/// `front` is an atomic so [`Reader::read`] can take `&self` (the public engine
/// API exposes `latest(&self)`). The consumer is still logically single-threaded;
/// the atomic is only for interior mutability, not cross-thread reader sharing.
pub struct Reader<T> {
    shared: Arc<Shared<T>>,
    /// Slot the consumer currently owns / last latched.
    front: AtomicU8,
}

/// Create a triple buffer initialized with `initial` in every slot.
pub fn triple_buffer<T: Copy>(initial: T) -> (Writer<T>, Reader<T>) {
    let shared = Arc::new(Shared {
        slots: [
            UnsafeCell::new(initial),
            UnsafeCell::new(initial),
            UnsafeCell::new(initial),
        ],
        // Published slot = 0, not dirty. Producer owns 1, consumer owns 2.
        ready: AtomicU8::new(0),
    });
    (
        Writer {
            shared: shared.clone(),
            back: 1,
        },
        Reader {
            shared,
            front: AtomicU8::new(2),
        },
    )
}

impl<T: Copy> Writer<T> {
    /// Publish a fresh value. Never blocks. Subsequent calls reuse a private slot,
    /// so the producer can write at any rate independent of the consumer.
    pub fn write(&mut self, value: T) {
        // SAFETY: `self.back` is owned exclusively by the producer until the swap below.
        unsafe {
            *self.shared.slots[self.back as usize].get() = value;
        }
        // Publish: swap our back index into `ready`, mark dirty. We receive the
        // previously published index, which becomes our new private back slot.
        let published = self.back | DIRTY_BIT;
        let prev = self.shared.ready.swap(published, Ordering::AcqRel);
        self.back = prev & INDEX_MASK;
    }
}

impl<T: Copy> Reader<T> {
    /// Read the most recently published value. Never blocks; if nothing new was
    /// published since the last call it returns the same latest value again.
    pub fn read(&self) -> T {
        // If dirty, latch the newly published slot as ours and clear the dirty bit.
        // We do this by swapping our front index into `ready` (clearing dirty) and
        // taking the published index as our new front.
        let ready = self.shared.ready.load(Ordering::Acquire);
        let mut front = self.front.load(Ordering::Relaxed);
        if ready & DIRTY_BIT != 0 {
            // Hand our current front slot back to the producer pool and adopt the
            // freshly published one.
            front = self.shared.ready.swap(front, Ordering::AcqRel) & INDEX_MASK;
            self.front.store(front, Ordering::Relaxed);
        }
        // SAFETY: `front` is owned exclusively by the consumer between swaps.
        unsafe { *self.shared.slots[front as usize].get() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_value_is_readable() {
        let (_w, r) = triple_buffer(7u32);
        assert_eq!(r.read(), 7);
        // Repeated reads with no writes return the same value.
        assert_eq!(r.read(), 7);
    }

    #[test]
    fn read_sees_latest_write() {
        let (mut w, r) = triple_buffer(0u32);
        w.write(1);
        assert_eq!(r.read(), 1);
        w.write(2);
        w.write(3);
        // Always the latest, intermediate values may be skipped.
        assert_eq!(r.read(), 3);
    }

    #[test]
    fn write_without_read_does_not_deadlock() {
        let (mut w, r) = triple_buffer(0u32);
        for i in 0..1000 {
            w.write(i);
        }
        assert_eq!(r.read(), 999);
    }
}
