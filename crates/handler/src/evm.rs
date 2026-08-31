use crate::{
    instructions::InstructionProvider, item_or_result::FrameInitOrResult, EthFrame, FrameResult,
    ItemOrResult, PrecompileProvider,
};
use auto_impl::auto_impl;
use context::{ContextTr, Database, Evm, FrameStack};
use context_interface::context::ContextError;
use interpreter::{interpreter::EthInterpreter, interpreter_action::FrameInit, InterpreterResult};

/// Type alias for database error within a context
pub type ContextDbError<CTX> = ContextError<ContextTrDbError<CTX>>;

/// Type alias for frame error within a context
pub type ContextTrDbError<CTX> = <<CTX as ContextTr>::Db as Database>::Error;

/// Type alias for frame init result
pub type FrameInitResult<'a, F> = ItemOrResult<&'a mut F, <F as FrameTr>::FrameResult>;

/// Trait for defining a frame type used in EVM execution.
#[auto_impl(&mut, Box)]
pub trait FrameTr {
    /// The result type returned when a frame completes execution.
    type FrameResult: From<FrameResult>;
    /// The initialization type used to create a new frame.
    type FrameInit: From<FrameInit>;
}

/// A trait that integrates context, instruction set, and precompiles to create an EVM struct.
///
/// In addition to execution capabilities, this trait provides getter methods for its component fields.
#[auto_impl(&mut, Box)]
pub trait EvmTr {
    /// The context type that implements ContextTr to provide access to execution state
    type Context: ContextTr;
    /// The instruction set type that implements InstructionProvider to define available operations
    type Instructions: InstructionProvider;
    /// The type containing the available precompiled contracts
    type Precompiles: PrecompileProvider<Self::Context>;
    /// The type containing the frame
    type Frame: FrameTr;

    /// Returns a tuple of references to the context, the frame and the instructions.
    #[allow(clippy::type_complexity)]
    fn all(
        &self,
    ) -> (
        &Self::Context,
        &Self::Instructions,
        &Self::Precompiles,
        &FrameStack<Self::Frame>,
    );

    /// Returns a tuple of mutable references to the context, the frame and the instructions.
    #[allow(clippy::type_complexity)]
    fn all_mut(
        &mut self,
    ) -> (
        &mut Self::Context,
        &mut Self::Instructions,
        &mut Self::Precompiles,
        &mut FrameStack<Self::Frame>,
    );

    /// Returns a mutable reference to the execution context
    #[inline]
    fn ctx(&mut self) -> &mut Self::Context {
        let (ctx, _, _, _) = self.all_mut();
        ctx
    }

    /// Returns a mutable reference to the execution context
    #[inline]
    fn ctx_mut(&mut self) -> &mut Self::Context {
        self.ctx()
    }

    /// Returns an immutable reference to the execution context
    #[inline]
    fn ctx_ref(&self) -> &Self::Context {
        let (ctx, _, _, _) = self.all();
        ctx
    }

    /// Returns mutable references to both the context and instruction set.
    /// This enables atomic access to both components when needed.
    #[inline]
    fn ctx_instructions(&mut self) -> (&mut Self::Context, &mut Self::Instructions) {
        let (ctx, instructions, _, _) = self.all_mut();
        (ctx, instructions)
    }

    /// Returns mutable references to both the context and precompiles.
    /// This enables atomic access to both components when needed.
    #[inline]
    fn ctx_precompiles(&mut self) -> (&mut Self::Context, &mut Self::Precompiles) {
        let (ctx, _, precompiles, _) = self.all_mut();
        (ctx, precompiles)
    }

    /// Returns a mutable reference to the frame stack.
    #[inline]
    fn frame_stack(&mut self) -> &mut FrameStack<Self::Frame> {
        let (_, _, _, frame_stack) = self.all_mut();
        frame_stack
    }

    /// Initializes the frame for the given frame input. Frame is pushed to the frame stack.
    fn frame_init(
        &mut self,
        frame_input: <Self::Frame as FrameTr>::FrameInit,
    ) -> Result<FrameInitResult<'_, Self::Frame>, ContextDbError<Self::Context>>;

    /// Run the frame from the top of the stack. Returns the frame init or result.
    ///
    /// If frame has returned result it would mark it as finished.
    fn frame_run(
        &mut self,
    ) -> Result<FrameInitOrResult<Self::Frame>, ContextDbError<Self::Context>>;

    /// Returns the result of the frame to the caller. Frame is popped from the frame stack.
    ///
    /// Takes the result by `&mut` and reports whether it was the outermost frame, rather
    /// than taking it by value and handing it back: a `FrameResult` is ~96 bytes with
    /// align-1 fields inside, so every by-value hand-off is a byte-wide stack copy.
    fn frame_return_result(
        &mut self,
        result: &mut <Self::Frame as FrameTr>::FrameResult,
    ) -> Result<bool, ContextDbError<Self::Context>>;
}

impl<CTX, INSP, I, P> EvmTr for Evm<CTX, INSP, I, P, EthFrame<EthInterpreter>>
where
    CTX: ContextTr,
    I: InstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    type Context = CTX;
    type Instructions = I;
    type Precompiles = P;
    type Frame = EthFrame<EthInterpreter>;

    #[inline]
    fn all(
        &self,
    ) -> (
        &Self::Context,
        &Self::Instructions,
        &Self::Precompiles,
        &FrameStack<Self::Frame>,
    ) {
        let ctx = &self.ctx;
        let instructions = &self.instruction;
        let precompiles = &self.precompiles;
        let frame_stack = &self.frame_stack;
        (ctx, instructions, precompiles, frame_stack)
    }

    #[inline]
    fn all_mut(
        &mut self,
    ) -> (
        &mut Self::Context,
        &mut Self::Instructions,
        &mut Self::Precompiles,
        &mut FrameStack<Self::Frame>,
    ) {
        let ctx = &mut self.ctx;
        let instructions = &mut self.instruction;
        let precompiles = &mut self.precompiles;
        let frame_stack = &mut self.frame_stack;
        (ctx, instructions, precompiles, frame_stack)
    }

    /// Initializes the frame for the given frame input. Frame is pushed to the frame stack.
    #[inline]
    fn frame_init(
        &mut self,
        frame_input: <Self::Frame as FrameTr>::FrameInit,
    ) -> Result<FrameInitResult<'_, Self::Frame>, ContextDbError<CTX>> {
        let is_first_init = self.frame_stack.index().is_none();
        let new_frame = if is_first_init {
            self.frame_stack.start_init()
        } else {
            self.frame_stack.get_next()
        };

        let ctx = &mut self.ctx;
        let precompiles = &mut self.precompiles;
        // Spelled out with `match` rather than `?` plus `map_frame`. `Try::branch` takes the
        // `Result` by value and hands the payload back inside a `ControlFlow`, and
        // `map_frame` takes `self` by value and rebuilds the enum around it; each is a real
        // copy of the ~96-byte `FrameResult` through a fresh stack slot, and since the
        // payload carries align-1 fields that copy is a `memcpy` libcall. The two of them
        // were 40,156 of the guest's 324,718 `memcpy` calls on mainnet block 24006677, at
        // ~74 retired instructions each.
        //
        // The one copy left is the unavoidable one: the returned type differs from
        // `init_with_context`'s in its item variant, so the result variant has to be moved
        // from one enum into the other.
        let token = match Self::Frame::init_with_context(new_frame, ctx, precompiles, frame_input) {
            Err(e) => return Err(e),
            Ok(ItemOrResult::Result(result)) => return Ok(ItemOrResult::Result(result)),
            Ok(ItemOrResult::Item(token)) => token,
        };

        if is_first_init {
            unsafe { self.frame_stack.end_init(token) };
        } else {
            unsafe { self.frame_stack.push(token) };
        }
        Ok(ItemOrResult::Item(self.frame_stack.get()))
    }

    /// Run the frame from the top of the stack. Returns the frame init or result.
    #[inline]
    fn frame_run(&mut self) -> Result<FrameInitOrResult<Self::Frame>, ContextDbError<CTX>> {
        let frame = self.frame_stack.get();
        let context = &mut self.ctx;
        let instructions = &mut self.instruction;

        let action = frame
            .interpreter
            .run_plain(instructions.instruction_table(), context);

        // A plain tail call, deliberately. Anything between the call and the return -
        // even just the `frame.set_finished(true)` that used to live here, which LLVM
        // cannot prove does not alias the return slot - blocks the call-slot
        // optimisation, and the ~104-byte `ItemOrResult` is then copied out of the
        // callee sret slot into this function own one. That copy is byte-wide (the
        // payload carries align-1 fields) and measured at 4.6 M retired instructions.
        // `process_next_action` sets the frame finished flag itself instead.
        frame.process_next_action(context, action)
    }

    /// Returns the result of the frame to the caller. Frame is popped from the frame stack.
    #[inline]
    fn frame_return_result(
        &mut self,
        result: &mut <Self::Frame as FrameTr>::FrameResult,
    ) -> Result<bool, ContextDbError<Self::Context>> {
        if self.frame_stack.get().is_finished() {
            self.frame_stack.pop();
        }
        if self.frame_stack.index().is_none() {
            return Ok(true);
        }
        self.frame_stack
            .get()
            .return_result::<_, ContextDbError<Self::Context>>(&mut self.ctx, result)?;
        Ok(false)
    }
}
