"""create notes table

Revision ID: 0001_create_notes_table
Revises:
Create Date: 2026-05-28 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_create_notes_table"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("notes")
