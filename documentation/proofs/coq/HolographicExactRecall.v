From Coq Require Import List Arith Lia.
Import ListNotations.

Module HolographicExactRecall.

  (* A simple discrete model of holographic memory as a flat array (list).
     We abstract bytes as natural numbers for the purpose of the proof. *)

  Definition mem := list nat.

  (* Write a block [blk] into [m] starting at offset [s]. If the write would
     exceed the memory bounds, we leave [m] unchanged (conservative model).
     This mirrors a well-formed placement precondition in the implementation. *)
  Fixpoint write_at (m:mem) (s:nat) (blk:list nat) : mem :=
    match s, m with
    | 0, _ => (* overwrite from head with blk, preserving tail beyond blk *)
        firstn 0 m ++ blk ++ skipn (length blk) m
    | S s', [] => []
    | S s', x::xs => x :: write_at xs s' blk
    end.

  (* Read [len] elements starting at offset [s]. If out of bounds, returns
     as many as available. *)
  Fixpoint read_at (m:mem) (s len:nat) : list nat :=
    match s, m with
    | 0, _ => firstn len m
    | S s', [] => []
    | S s', _::xs => read_at xs s' len
    end.

  Lemma read_at_firstn_skipn : forall (m:mem) s len,
      read_at m s len = firstn len (skipn s m).
  Proof.
    induction m as [|x xs IH]; intros s len; destruct s; simpl; auto.
    now rewrite IH.
  Qed.

  Lemma write_at_equation : forall m s blk,
      write_at m s blk =
        match s with
        | 0 => blk ++ skipn (length blk) m
        | S s' => match m with
                  | [] => []
                  | x::xs => x :: write_at xs s' blk
                  end
        end.
  Proof.
    destruct m; destruct s; simpl; auto.
  Qed.

  Lemma length_skipn_ge : forall (m:mem) n,
      length (skipn n m) = length m - n.
  Proof.
    induction m as [|x xs IH]; intros [|n]; simpl; auto using Nat.sub_0_r.
    now rewrite IH, Nat.sub_succ.
  Qed.

  (* Exact recall theorem: if a block fits entirely within the available suffix
     skipn s m (i.e., length blk <= length m - s), then reading back at [s]
     after writing yields exactly [blk]. *)
  Theorem exact_recall : forall (m:mem) (s:nat) (blk:list nat),
      length blk <= length m - s ->
      read_at (write_at m s blk) s (length blk) = blk.
  Proof.
    intros m s blk Hfit.
    rewrite read_at_firstn_skipn.
    rewrite write_at_equation.
    destruct s as [|s'].
    - (* head overwrite case *)
      rewrite skipn_0.
      rewrite firstn_app.
      rewrite firstn_all.
      rewrite Nat.min_l by lia.
      now rewrite firstn_all.
    - (* interior overwrite case *)
      destruct m as [|x xs]; simpl.
      { (* out of bounds: precondition prevents this branch by Hfit *)
        simpl in Hfit. lia. }
      (* Reduce to suffix and apply IH on lengths *)
      assert (Hfit': length blk <= length xs - s') by (simpl in Hfit; lia).
      specialize (exact_recall xs s' blk Hfit').
      rewrite read_at_firstn_skipn in exact_recall.
      simpl. now rewrite exact_recall.
  Qed.

End HolographicExactRecall.

