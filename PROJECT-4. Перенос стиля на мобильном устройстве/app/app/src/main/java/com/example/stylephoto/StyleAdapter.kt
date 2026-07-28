package com.example.stylephoto

import android.graphics.Bitmap
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.RecyclerView

class StyleAdapter(
    private val styles: List<Bitmap>,
    private val onStyleSelected: (Int) -> Unit
) : RecyclerView.Adapter<StyleAdapter.StyleViewHolder>() {

    private var selectedIndex = RecyclerView.NO_ID.toInt()

    inner class StyleViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val imageView: ImageView = view.findViewById(R.id.styleImage)
        val selectionRing: View = view.findViewById(R.id.selectionRing)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): StyleViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_style, parent, false)
        return StyleViewHolder(view)
    }

    override fun onBindViewHolder(holder: StyleViewHolder, position: Int) {
        holder.imageView.setImageBitmap(styles[position])

        val isSelected = position == selectedIndex
        holder.imageView.alpha = if (isSelected) 1f else 0.55f
        holder.selectionRing.visibility = if (isSelected) View.VISIBLE else View.GONE

        holder.itemView.setOnClickListener {
            val prev = selectedIndex
            selectedIndex = holder.adapterPosition     // use adapterPosition, not captured position
            notifyItemChanged(prev)
            notifyItemChanged(selectedIndex)
            onStyleSelected(selectedIndex)
        }
    }

    override fun getItemCount() = styles.size
}